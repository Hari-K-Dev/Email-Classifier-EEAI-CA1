# Design Choice 2: Hierarchical Modelling
#
# Runs ALL three models (RandomForest, LogisticRegression, SVM) at every
# level of the hierarchy, with a comparison table + 5-fold CV per node.
#
# Architecture:
#   Level 1 - All 3 models classify y2 on ALL data
#   Level 2 - For each y2 class, all 3 models classify y3 on filtered subset
#   Level 3 - For each (y2, y3) pair, all 3 models classify y4 on further-filtered subset

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from Config import Config
from modelling.data_model import Data

from model.randomforest import RandomForest
from model.logistic_regression import LogisticRegression
from model.svm import SVM

MODEL_CLASSES = [RandomForest, LogisticRegression, SVM]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_enough_data(y_train: np.ndarray, y_test: np.ndarray,
                     min_train: int = 2, min_test: int = 1) -> bool:
    if len(y_train) < min_train or len(y_test) < min_test:
        return False
    if len(np.unique(y_train)) < 2:
        return False
    return True


def _cross_validate(model_cls, X: np.ndarray, y: np.ndarray,
                    n_splits: int = 5) -> float:
    """Stratified K-Fold CV — returns mean accuracy, or nan if too few samples."""
    counts = Counter(y)
    actual_splits = min(n_splits, min(counts.values()))
    if actual_splits < 2:
        return float('nan')

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True,
                          random_state=Config.SEED)
    scores = []
    for tr_idx, te_idx in skf.split(X, y):
        mdl = model_cls(model_name='cv', embeddings=X[tr_idx], y=y[tr_idx])

        class _D:
            pass
        d = _D()
        d.X_train = X[tr_idx]
        d.y_train = y[tr_idx]

        mdl.train(d)
        mdl.predict(X[te_idx])
        scores.append(accuracy_score(y[te_idx], mdl.predictions))

    return float(np.mean(scores))


def _run_all_models(X_train, y_train, X_test, y_test, indent: str = "") -> None:
    """Train + evaluate all 3 models, print per-model report and comparison table with CV."""
    results = {}
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    for model_cls in MODEL_CLASSES:
        label = model_cls.__name__
        print(f"\n{indent}  -- {label} --")

        mdl = model_cls(model_name=f"{label}_hier", embeddings=X_train, y=y_train)

        class _Data:
            pass
        d = _Data()
        d.X_train = X_train
        d.y_train = y_train
        d.X_test  = X_test
        d.y_test  = y_test

        mdl.train(d)
        mdl.predict(X_test)
        print(f"{indent}  {classification_report(y_test, mdl.predictions, zero_division=0)}")

        hold_acc = accuracy_score(y_test, mdl.predictions)
        cv_acc   = _cross_validate(model_cls, X_all, y_all)
        results[label] = (hold_acc, cv_acc)

    # Comparison table for this node
    print(f"{indent}  {'─'*54}")
    print(f"{indent}  {'Model':<22} {'Hold-out Acc':>13} {'5-Fold CV Acc':>14}")
    print(f"{indent}  {'─'*54}")
    for lbl, (hold_acc, cv_acc) in results.items():
        cv_str = f"{cv_acc:.2%}" if not np.isnan(cv_acc) else "n/a"
        print(f"{indent}  {lbl:<22} {hold_acc:>12.2%} {cv_str:>14}")
    print(f"{indent}  {'─'*54}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def hierarchical_model_predict(X: np.ndarray, df: pd.DataFrame) -> None:
    """Run Design Choice 2 — Hierarchical Modelling with all 3 models + CV."""

    # -----------------------------------------------------------------------
    # LEVEL 1 — Classify y2 on ALL data
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LEVEL 1  |  Classifying y2  |  ALL data")
    print(f"{'='*70}")

    data_l1 = Data(X, df, target_col='y2')
    _run_all_models(data_l1.X_train, data_l1.y_train,
                    data_l1.X_test,  data_l1.y_test)

    train_df     = data_l1.train_df.copy().reset_index(drop=True)
    test_df      = data_l1.test_df.copy().reset_index(drop=True)
    X_train_full = data_l1.X_train
    X_test_full  = data_l1.X_test

    # -----------------------------------------------------------------------
    # LEVEL 2 — Per y2 class: filter → classify y3
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LEVEL 2  |  Classifying y3  |  Filtered per y2 class")
    print(f"{'='*70}")

    for y2_cls in sorted(train_df['y2'].unique()):
        tr_mask = train_df['y2'] == y2_cls
        te_mask = test_df['y2']  == y2_cls

        X_tr2 = X_train_full[tr_mask.values]
        X_te2 = X_test_full[te_mask.values]
        y_tr2 = train_df.loc[tr_mask, 'y3'].values
        y_te2 = test_df.loc[te_mask,  'y3'].values

        print(f"\n  -- y2 = '{y2_cls}'  (train: {len(y_tr2)}, test: {len(y_te2)}) --")

        if not _has_enough_data(y_tr2, y_te2):
            print("     [skipped — not enough samples or only one y3 class]")
            continue

        _run_all_models(X_tr2, y_tr2, X_te2, y_te2, indent="  ")

        # -------------------------------------------------------------------
        # LEVEL 3 — Per y3 class within this y2 subset: filter → classify y4
        # -------------------------------------------------------------------
        print(f"\n{'─'*70}")
        print(f"  LEVEL 3  |  Classifying y4  |  y2='{y2_cls}' → filtered by y3")
        print(f"{'─'*70}")

        sub_train = train_df[tr_mask].reset_index(drop=True)
        sub_test  = test_df[te_mask].reset_index(drop=True)

        for y3_cls in sorted(sub_train['y3'].unique()):
            tr3_mask = sub_train['y3'] == y3_cls
            te3_mask = sub_test['y3']  == y3_cls

            X_tr3 = X_tr2[tr3_mask.values]
            X_te3 = X_te2[te3_mask.values]
            y_tr3 = sub_train.loc[tr3_mask, 'y4'].values
            y_te3 = sub_test.loc[te3_mask,  'y4'].values

            print(f"\n    -- y2='{y2_cls}', y3='{y3_cls}'  "
                  f"(train: {len(y_tr3)}, test: {len(y_te3)}) --")

            if not _has_enough_data(y_tr3, y_te3):
                print("       [skipped — not enough samples or only one y4 class]")
                continue

            _run_all_models(X_tr3, y_tr3, X_te3, y_te3, indent="    ")
