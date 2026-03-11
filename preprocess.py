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
    """Clean text columns: remove emails, URLs, HTML entities, special chars, extra whitespace."""
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = str(text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Replace HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        # Remove special characters (keep letters, numbers, spaces, basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].apply(clean_text)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(clean_text)

    # Fill NaN in type columns with 'Unknown'
    for col in Config.TYPE_COLS:
        df[col] = df[col].fillna('Unknown')

    return df


def translate_to_en(texts: list) -> list:
    """Translate non-English text to English using deep_translator, with fallback to passthrough."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        translated = []
        for text in texts:
            if not text or text.strip() == '':
                translated.append(text)
                continue
            try:
                result = translator.translate(text)
                translated.append(result if result else text)
            except Exception:
                translated.append(text)
        return translated
    except ImportError:
        # Fallback: no translation library available, return as-is
        return texts
