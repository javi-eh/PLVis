import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def compute_embeddings(df=None):
    embeddings_file = 'data/embeddings.npy'
    model = SentenceTransformer('sentence-transformers/gtr-t5-xl')

    if df is not None:
        abstracts = df['Abstracts'].fillna('').tolist()
        embeddings = model.encode(abstracts, show_progress_bar=True)
        return embeddings
    else:
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
        else:
            df = pd.read_csv('data/mtuberculosis_df_abs.csv')
            abstracts = df['Abstracts'].fillna('').tolist()
            embeddings = model.encode(abstracts, show_progress_bar=True)
            np.save(embeddings_file, embeddings)
        return embeddings
