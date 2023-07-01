from abc import ABC, abstractmethod


class GenericForecaster(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @abstractmethod
    def predict(self, *args, **kwargs):
        ...