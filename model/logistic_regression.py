# Logistic Regression model — extends BaseModel
# Demonstrates that the abstraction layer works: this model plugs in
# without any changes to main_both.py, modelling.py, or data_model.py.
# It only needs to implement train(), predict(), print_results(), data_transform().

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import classification_report
from model.base import BaseModel
from Config import Config

import random
random.seed(Config.SEED)
np.random.seed(Config.SEED)


class LogisticRegression(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        # max_iter=1000 avoids convergence warnings on small datasets
        # class_weight='balanced' handles class imbalance same as RF does
        self.mdl = SklearnLR(
            max_iter=1000,
            random_state=Config.SEED,
            class_weight='balanced',
            solver='lbfgs'
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
