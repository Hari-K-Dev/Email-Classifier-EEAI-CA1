#Methods related to converting text in into numeric representation and then returning numeric representation may go here
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """Create TF-IDF embeddings from Ticket Summary and Interaction content combined."""
    # Combine both text columns into a single corpus
    corpus = (
        df[Config.TICKET_SUMMARY].astype(str) + ' ' +
        df[Config.INTERACTION_CONTENT].astype(str)
    )

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    X = vectorizer.fit_transform(corpus)
    return X.toarray()  # Convert sparse to dense for RandomForest compatibility
