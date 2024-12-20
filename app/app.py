
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from embeddings import compute_embeddings
from chatbot import get_chatbot_response
from sklearn.decomposition import PCA
import re

app = Flask(__name__)

# Global variables to store user selections
sel_col = 'Total_Counts'
sel_comp = 'All proteomes'
plot_type = '3D'  # default plot type

@app.route('/', methods=['GET', 'POST'])
def index():
    global sel_col, sel_comp, plot_type

    if request.method == 'POST':
        # Get user selections from the form
        sel_col = request.form.get('sel_col')
        sel_comp = request.form.get('sel_comp')
        plot_type = request.form.get('plot_type', '3D')

    def clean_text(text):
        text = re.sub(r'\d+', '', str(text))
        # Remove parentheses and content within them
        while '(' in text and ')' in text:
            text = re.sub(r'\([^()]*\)', '', text)
        return text.strip()

    def normalize_columns(df, columns):
        for column_name in columns:
            min_val = df[column_name].min()
            max_val = df[column_name].max()

            if max_val - min_val == 0:
                df[column_name + '_normalized'] = 0
            else:
                df[column_name + '_normalized'] = (df[column_name] - min_val) / (max_val - min_val)
        return df

    print("LOADING DATAFRAME...")
    mycobacterium_df = pd.read_csv('data/mtuberculosis_df_abs.csv')
    mycobacterium_df['Organism'] = mycobacterium_df['Organism'].apply(clean_text)
    organisms = mycobacterium_df['Organism'].unique()

    # Assign colors to organisms
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_mapping = {organism: color_palette[i % len(color_palette)] for i, organism in enumerate(organisms)}
    mycobacterium_df['Color'] = mycobacterium_df['Organism'].map(color_mapping)

    print("LOADING COUNTS ALL STAGES...")
    counts_df = pd.read_excel('data/counts_all_stages_MAGECK_with_ES.xlsx')
    counts_df['clean_orf'] = counts_df['orf'].apply(lambda x: re.sub(r'(?<=RV)BD', '', str(x), count=1))
    counts_df['clean_name'] = counts_df['name'].apply(lambda x: re.sub(r'(?<=RV)BD', '', str(x), count=1))

    tb_df = mycobacterium_df[mycobacterium_df['Organism'] == 'Mycobacterium tuberculosis'].copy()
    cols = ['Counts_1st', 'Counts_2nd', 'Counts_3rd', 'Total_Counts', 'Rank']
    tb_df[cols] = np.nan

    for _, row in counts_df.iterrows():
        clean_orf_escaped = re.escape(str(row['clean_orf']))
        clean_name_escaped = re.escape(str(row['clean_name']))
        pattern = f"{clean_orf_escaped}|{clean_name_escaped}"
        mask = tb_df['Gene Names'].str.contains(pattern, case=False, na=False, regex=True)
        tb_df.loc[mask, cols] = [row['Counts_1st'], row['Counts_2nd'], row['Counts_3rd'], row['Total_Counts'], row['Rank']]

    tb_df = normalize_columns(tb_df, cols)

    # Ensure sel_col is normalized column if original not found
    if sel_col not in cols and not sel_col.endswith('_normalized'):
        sel_col = 'Total_Counts_normalized'

    tb_df_subset = tb_df[['Entry', sel_col, 'Rank']]

    if sel_comp == 'All proteomes':
        mycobacterium_df_subset = mycobacterium_df.copy()
        plot_df = pd.merge(mycobacterium_df_subset, tb_df_subset, how='left', on='Entry')
        plot_df[sel_col] = plot_df[sel_col].fillna(0)
    elif sel_comp == 'Mycobacterium tuberculosis':
        plot_df = tb_df.copy()
        plot_df[sel_col] = plot_df[sel_col].fillna(0)
    else:
        other_org = sel_comp[3:]
        mycobacterium_df_subset = mycobacterium_df[
            (mycobacterium_df['Organism'] == 'Mycobacterium tuberculosis') |
            (mycobacterium_df['Organism'].str.contains(other_org, case=False, na=False))
        ]
        plot_df = pd.merge(mycobacterium_df_subset, tb_df_subset, how='left', on='Entry')
        plot_df[sel_col] = plot_df[sel_col].fillna(0)

    # Compute point sizes
    min_size = 10
    max_size = 50
    plot_df['Size'] = min_size + (plot_df[sel_col] * (max_size - min_size))

    print("COMPUTING EMBEDDINGS...")
    embeddings = compute_embeddings()

    coords = None
    # Check if plot_type is "3D PCA Based" or just '3D'
    # To keep backward compatibility, let's treat '3D PCA Based' or '3D' the same.
    if plot_type.startswith('3D'):
        print("REDUCING DIMENSIONS (PCA)...")
        coords_file = 'data/coordinates.npy'
        if os.path.exists(coords_file):
            coords = np.load(coords_file)
        else:
            pca = PCA(n_components=3)
            coords = pca.fit_transform(embeddings)
            np.save(coords_file, coords)

    nodes = []
    for i in range(len(plot_df)):
        row = plot_df.iloc[i]
        protein = row.get('Protein names', 'N/A')
        organism = row.get('Organism', 'N/A')
        gene = row.get('Gene Names', 'N/A')
        pathway = row.get('Pathway', 'N/A')
        anot = row.get('Annotation', 'N/A')
        counts_val = row.get(sel_col, 'N/A')
        size = row.get('Size', 10)
        label = row.get('Cluster Label', 'N/A')
        color = row.get('Color', '#1f77b4')

        text = (
            f"Protein Names: {protein}<br>Organism: {organism}<br>Gene Names: {gene}<br>"
            f"Pathway: {pathway}<br>Counts: {counts_val}<br>Annotation: {anot}<br>"
            f"Point Size: {size}<br>Cluster: {label}"
        )

        if plot_type.startswith('3D'):
            # Use PCA coords
            x, y, z = coords[i]
            node = {
                'id': int(i),
                'protein_name': protein,
                'label': text,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'group': str(label),
                'size': float(size),
                'color': color,
            }
        else:
            # Use UMAP 1 and UMAP 2 for 2D plot
            x2d = row.get('UMAP 1', 0.0)
            y2d = row.get('UMAP 2', 0.0)
            node = {
                'id': int(i),
                'protein_name': protein,
                'label': text,
                'x': float(x2d),
                'y': float(y2d),
                'group': str(label),
                'size': float(size),
                'color': color,
            }

        nodes.append(node)

    edges = []  # No edges computed in this scenario

    column_options = ['Counts_1st_normalized', 'Counts_2nd_normalized', 'Counts_3rd_normalized', 'Total_Counts_normalized', 'Rank_normalized']
    comparison_options = ['All proteomes', 'Mycobacterium tuberculosis', 'vs smegmatis', 'vs marinum', 'vs leprae', 'vs kansasii', 'vs intracellulare', 'vs fortuitum', 'vs bovis']
    plot_options = ['3D PCA Based', '2D UMAP Based']

    return render_template('index.html',
                           nodes=nodes,
                           edges=edges,
                           sel_col=sel_col,
                           sel_comp=sel_comp,
                           column_options=column_options,
                           comparison_options=comparison_options,
                           plot_type=plot_type,
                           plot_options=plot_options)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    node_ids = data.get('node_ids', [])
    message = data.get('message', '')
    include_similar = data.get('include_similar', True)
    response_text = get_chatbot_response(node_ids, message, include_similar)
    return jsonify({'message': response_text})

if __name__ == '__main__':
    app.run(debug=True)
