# Support Vector Machine model — extends BaseModel
# SVM is well-suited for high-dimensional sparse text features (TF-IDF).
# LinearSVC is used instead of SVC(kernel='rbf') because:
#   - Much faster on text data
#   - Scales better with feature count (500 TF-IDF features here)
#   - Generally outperforms RBF kernel on linear text classification tasks

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from model.base import BaseModel
from Config import Config

import random
random.seed(Config.SEED)
np.random.seed(Config.SEED)


class SVM(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        # max_iter=2000 prevents convergence warnings on chained label levels
        # class_weight='balanced' compensates for unequal class sizes
        self.mdl = LinearSVC(
            max_iter=2000,
            random_state=Config.SEED,
            class_weight='balanced',
            C=1.0
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: np.ndarray) -> None:
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data) -> None:
        print(classification_report(data.y_test, self.predictions, zero_division=0))

    def data_transform(self) -> None:
        ...
