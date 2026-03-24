# Design Choice 1: Chained Multi-Output Classification
#
# Runs ALL three models (RandomForest, LogisticRegression, SVM) across
# every chained level. This directly demonstrates that the BaseModel
# abstraction works — new models plug in without touching any other file.
#
# Also runs Stratified K-Fold cross-validation (k=5) per model per level
# so that accuracy estimates are not dependent on a single 80/20 split.

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from collections import Counter

from model.randomforest import RandomForest
from model.logistic_regression import LogisticRegression
from model.svm import SVM
from modelling.data_model import Data
from Config import Config
from utils import create_chained_label


# ---------------------------------------------------------------------------
# Model registry — add new models here, nothing else needs to change
# ---------------------------------------------------------------------------
MODEL_CLASSES = [RandomForest, LogisticRegression, SVM]


# ---------------------------------------------------------------------------
# K-Fold cross-validation helper
# ---------------------------------------------------------------------------
def cross_validate(model_cls, X: np.ndarray, y: np.ndarray,
                   n_splits: int = 5) -> float:
    """Run StratifiedKFold CV and return mean accuracy across folds."""
    counts = Counter(y)
    min_count = min(counts.values())
    actual_splits = min(n_splits, min_count)

    if actual_splits < 2:
        return float('nan')

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True,
                          random_state=Config.SEED)
    fold_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        mdl = model_cls(model_name='cv_fold', embeddings=X_tr, y=y_tr)

        class _FoldData:
            pass
        fold_data = _FoldData()
        fold_data.X_train = X_tr
        fold_data.y_train = y_tr

        mdl.train(fold_data)
        mdl.predict(X_te)
        fold_scores.append(accuracy_score(y_te, mdl.predictions))

    return float(np.mean(fold_scores))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def model_predict(data, df, name):
    """Run Design Choice 1 across all models and all chained levels."""
    X = data.get_embeddings()
    summary = {}

    for i, level_cols in enumerate(Config.CHAINED_LEVELS):
        level_name = Config.CHAIN_SEP.join(level_cols)
        print(f"\n{'='*60}")
        print(f"Level {i+1}: Classifying {level_name}")
        print(f"{'='*60}")

        chained_col_name = '_'.join(level_cols)
        df[chained_col_name] = create_chained_label(df, level_cols, Config.CHAIN_SEP)
        level_data = Data(X, df, target_col=chained_col_name)

        level_summary = {}
        X_all = np.vstack([level_data.X_train, level_data.X_test])
        y_all = np.concatenate([level_data.y_train, level_data.y_test])

        for model_cls in MODEL_CLASSES:
            model_label = model_cls.__name__
            print(f"\n  -- {model_label} --")

            mdl = model_cls(
                model_name=f"{model_label}_Level{i+1}",
                embeddings=level_data.get_embeddings(),
                y=level_data.get_type()
            )
            mdl.train(level_data)
            mdl.predict(level_data.get_X_test())
            model_evaluate(mdl, level_data)

            hold_acc = accuracy_score(level_data.y_test, mdl.predictions)
            cv_acc = cross_validate(model_cls, X_all, y_all)
            level_summary[model_label] = (hold_acc, cv_acc)

        summary[f"Level {i+1} ({level_name})"] = level_summary
        _print_level_table(level_name, level_summary)

    _print_final_summary(summary)


def model_evaluate(model, data):
    model.print_results(data)


def _print_level_table(level_name: str, level_summary: dict) -> None:
    print(f"\n  {'─'*54}")
    print(f"  Comparison — {level_name}")
    print(f"  {'─'*54}")
    print(f"  {'Model':<22} {'Hold-out Acc':>13} {'5-Fold CV Acc':>14}")
    print(f"  {'─'*54}")
    for model_name, (hold_acc, cv_acc) in level_summary.items():
        cv_str = f"{cv_acc:.2%}" if not np.isnan(cv_acc) else "   n/a"
        print(f"  {model_name:<22} {hold_acc:>12.2%} {cv_str:>14}")
    print(f"  {'─'*54}")


def _print_final_summary(summary: dict) -> None:
    model_names = list(next(iter(summary.values())).keys())
    col_w = 20

    print(f"\n\n{'='*72}")
    print("DESIGN CHOICE 1 — FINAL MODEL COMPARISON ACROSS ALL LEVELS")
    print(f"{'='*72}")
    header = f"{'Level':<28}" + "".join(f"{m:>{col_w}}" for m in model_names)
    print(header)
    print("─" * len(header))
    for level, models in summary.items():
        row = f"{level:<28}"
        for m in model_names:
            hold_acc, _ = models[m]
            row += f"{hold_acc:>{col_w}.2%}"
        print(row)
    print()
    print("  Hold-out accuracy on fixed 20% test split.")
    print(f"{'='*72}")
