# main_both.py — Controller for BOTH Design Choices
#
# Design Choice 1: Chained Multi-Output Classification
#   One RF instance per level, target label is a concatenated string:
#     Level 1 → y2
#     Level 2 → y2 + y3
#     Level 3 → y2 + y3 + y4
#
# Design Choice 2: Hierarchical Modelling
#   Multiple RF instances, each trained on a filtered subset:
#     Level 1 → RF on all data, classifies y2
#     Level 2 → One RF per y2 class, classifies y3
#     Level 3 → One RF per (y2, y3) pair, classifies y4

import random
import numpy as np
from preprocess import get_input_data, de_duplication, noise_remover, translate_to_en
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.modelling import model_predict          # Design Choice 1
from modelling.hierarchical_modelling import hierarchical_model_predict  # Design Choice 2
from Config import Config

# ── Reproducibility ─────────────────────────────────────────────────────────
random.seed(Config.SEED)
np.random.seed(Config.SEED)


# ── Pipeline steps ───────────────────────────────────────────────────────────

def load_data():
    """Load and concatenate all CSV data files."""
    return get_input_data()


def preprocess_data(df):
    """Deduplicate, clean noise, and translate to English."""
    df = de_duplication(df)
    df = noise_remover(df)
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df):
    """Build TF-IDF feature matrix from text columns."""
    X = get_tfidf_embd(df)
    return X


def run_design_choice_1(X, df):
    """Design Choice 1 — Chained Multi-Output Classification."""
    print("\n")
    print("=" * 70)
    print("        DESIGN CHOICE 1 — CHAINED MULTI-OUTPUT")
    print("=" * 70)

    data = Data(X, df)
    model_predict(data, df, 'RandomForest')


def run_design_choice_2(X, df):
    """Design Choice 2 — Hierarchical Modelling."""
    print("\n")
    print("=" * 70)
    print("        DESIGN CHOICE 2 — HIERARCHICAL MODELLING")
    print("=" * 70)

    hierarchical_model_predict(X, df)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Pre-processing ───────────────────────────────────────────────────────
    print("\n[1/3] Loading and preprocessing data...")
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY]      = df[Config.TICKET_SUMMARY].values.astype('U')
    print(f"      Loaded {len(df)} rows after preprocessing.")

    # ── Embeddings ───────────────────────────────────────────────────────────
    print("\n[2/3] Building TF-IDF embeddings...")
    X = get_embeddings(df)
    print(f"      Feature matrix shape: {X.shape}")

    # ── Modelling ────────────────────────────────────────────────────────────
    print("\n[3/3] Running both design choices...\n")

    run_design_choice_1(X, df)

    print("\n" + "="*70 + "\n")

    run_design_choice_2(X, df)

    print("\n\nDone.")
