import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import markdown  

def get_chatbot_response(node_ids, message, include_similar=True, info_source='Abstracts'):
    df = pd.read_csv('data/mtuberculosis_df_abs.csv')

    if info_source == 'Abstracts':
        embeddings_file = 'data/embeddings_abs.npy'
        text_column = 'Abstracts'
    else:
        embeddings_file = 'data/embeddings_cc.npy'
        text_column = 'Function [CC]'

    embeddings = np.load(embeddings_file)

    # Ensure embeddings shape
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device='cpu')

    context = ""
    for node_id in node_ids:
        clicked_protein_name = df.iloc[node_id]['Protein names']
        clicked_protein_info = df.iloc[node_id][text_column]
        context += f"Selected Protein Name: {clicked_protein_name}\n"
        context += f"{info_source}: {clicked_protein_info}\n\n"

        if include_similar:
            clicked_protein_embedding = embeddings[node_id].reshape(1, -1)
            similarities = cosine_similarity(clicked_protein_embedding, embeddings)[0]

            K = 6
            top_k_indices = similarities.argsort()[-K:][::-1]
            top_k_indices = [idx for idx in top_k_indices if idx != node_id][:5]

            for idx in top_k_indices:
                protein_name = df.iloc[idx]['Protein names']
                related_info = df.iloc[idx][text_column]
                context += f"Related Protein Name: {protein_name}\n{info_source}: {related_info}\n\n"

    prompt = f"{context}\nUser Question: {message}\nAssistant:"

    openai.api_key = os.getenv('OPENAI_API_KEY')
    if openai.api_key is None:
        raise ValueError("OpenAI API key not set.")

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert assistant providing detailed information about proteins. "
                "You are being provided a list of proteins along with functional annotations (Context) from UniProt. "
                "Use that context plus any background knowledge to give an overview of what these proteins are. "
                "It is particularly useful to highlight commonalities and differences in these protein sets whenever asked to explain. "
                "The main goal is to orient the user as to what these proteins are, their function, and what is known about their biology."
            )},
            {"role": "user", "content": prompt}
        ]
    )

    assistant_reply = response['choices'][0]['message']['content']
    # Convert Markdown to HTML
    assistant_html = markdown.markdown(assistant_reply, extensions=['extra', 'nl2br'])
    return assistant_html

