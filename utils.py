#Any extra functionality that need to be reused will go here
import pandas as pd
import numpy as np


def remove_rare_classes(df: pd.DataFrame, col: str, threshold: int = 1) -> pd.DataFrame:
    """Remove rows belonging to classes with count <= threshold in the given column."""
    counts = df[col].value_counts()
    rare_classes = counts[counts <= threshold].index.tolist()
    df_filtered = df[~df[col].isin(rare_classes)]
    return df_filtered


def create_chained_label(df: pd.DataFrame, cols: list, sep: str = ' + ') -> pd.Series:
    """Create a chained label by concatenating multiple column values with a separator.
    E.g., cols=['y2','y3'] -> 'Suggestion + Payment'
    """
    result = df[cols[0]].astype(str)
    for col in cols[1:]:
        result = result + sep + df[col].astype(str)
    return result
