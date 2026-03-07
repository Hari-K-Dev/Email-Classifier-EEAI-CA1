#Methods related to data loading and all pre-processing steps will go here
import pandas as pd
import numpy as np
import re
from Config import Config


def get_input_data() -> pd.DataFrame:
    """Load both CSV files, concatenate, and rename type columns to internal names."""
    dfs = []
    for path in Config.DATA_FILES:
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    # Rename CSV columns to internal names: 'Type 2' -> 'y2', etc.
    rename_map = {v: k for k, v in Config.TYPE_COL_MAP.items()}
    df.rename(columns=rename_map, inplace=True)

    # Drop unnamed/empty trailing columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    return df


def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on Ticket id and Interaction id."""
    df = df.drop_duplicates(subset=['Ticket id', 'Interaction id'], keep='first')
    df = df.reset_index(drop=True)
    return df


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: implement noise removal
    return df


def translate_to_en(texts: list) -> list:
    # TODO: implement translation
    return texts
