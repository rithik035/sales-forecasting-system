from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def predict(self, steps):
        pass

    @abstractmethod
    def evaluate(self, test_data):
        pass
