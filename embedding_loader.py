import pandas as pd
import numpy as np
from config import DATA_PATH, EMBEDDINGS_PATH

def load_data_and_embeddings():
    df = pd.read_csv(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    df['embeddings'] = list(embeddings)
    return df
