from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self) -> None:
        ...


    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self) -> int:
        """
        Make predictions on test data.
        """
        ...

    @abstractmethod
    def print_results(self, data) -> None:
        """
        Print classification report for model evaluation.
        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        return

    def build(self, values={}):
        values = values if isinstance(values, dict) else {}
        self.__dict__.update(values)
        return self
