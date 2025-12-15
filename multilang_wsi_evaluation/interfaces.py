import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Sample:
    context: str
    begin: int
    end: int
    lemma: str
    lang: str


class IWSIVectorizer(ABC):
    @abstractmethod
    def predict(self, samples: tp.List[Sample]) -> tp.Any:
        """
        Args:
            samples: list of Samples for only one word
        Returns:
             Distance matrix or vectors of Samples
        """
        pass

    def fit(self, samples: tp.List[Sample]) -> None:
        """Compute some statistics etc.
        Args:
            samples: list of Samples for all words from one dataset
        """
        pass


class IWSI(ABC):
    @abstractmethod
    def predict(self, samples: tp.List[Sample]) -> tp.List[tp.Any]:
        """
        Args:
            samples: list of Samples for all words from one dataset
        Returns:
             List of clusters for the samples
        """
        pass
