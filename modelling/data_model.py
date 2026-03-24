import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
from utils import remove_rare_classes
import random
from collections import Counter

seed = 0
random.seed(seed)
np.random.seed(seed)


class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 target_col: str = None) -> None:
        """Build train/test split for a given target column.
        If target_col is None, defaults to Config.CLASS_COL ('y2').
        Removes rare classes, splits 80/20, stores all attributes.
        """
        if target_col is None:
            target_col = Config.CLASS_COL

        self.target_col = target_col

        # Store original embeddings (full matrix before filtering)
        self.embeddings = X

        # Remove rare classes
        df_clean = remove_rare_classes(df.copy(), target_col, threshold=Config.RARE_THRESHOLD)

        # Save original row indices into X BEFORE resetting the index
        # (after reset_index, indices become 0,1,2,... which would wrongly slice X)
        orig_indices = df_clean.index.tolist()
        X_clean = X[orig_indices]

        df_clean = df_clean.reset_index(drop=True)

        # Extract target
        self.y = df_clean[target_col].values

        # Check if stratified split is possible
        counts = Counter(self.y)
        can_stratify = all(c >= 2 for c in counts.values())

        # Train/test split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test, train_idx, test_idx = \
            train_test_split(
                X_clean, self.y,
                np.arange(len(df_clean)),
                test_size=0.2,
                random_state=seed,
                stratify=self.y if can_stratify else None
            )

        self.train_df = df_clean.iloc[train_idx].reset_index(drop=True)
        self.test_df = df_clean.iloc[test_idx].reset_index(drop=True)

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df
