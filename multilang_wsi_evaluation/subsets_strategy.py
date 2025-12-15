import typing as tp
from abc import ABC, abstractmethod

from multilang_wsi_evaluation.utils import powerset


class SubsetsStrategy(ABC):
    @abstractmethod
    def select_subsets(self, values: tp.List[tp.Any]) -> tp.List[tp.List[tp.Any]]:
        pass


class FullySharedStrategy(SubsetsStrategy):
    def select_subsets(self, values: tp.List[tp.Any]) -> tp.List[tp.List[tp.Any]]:
        return [[val] for val in values]


class FullyIndividualStrategy(SubsetsStrategy):
    def select_subsets(self, values: tp.List[tp.Any]) -> tp.List[tp.List[tp.Any]]:
        return [values]


class AllIntervalsStrategy(SubsetsStrategy):
    def select_subsets(self, values: tp.List[tp.Any]) -> tp.List[tp.List[tp.Any]]:
        subsets = []
        for ind_1 in range(len(values)):
            for ind_2 in range(ind_1 + 1, len(values) + 1):
                subsets.append(values[ind_1:ind_2])
        return subsets


class AllSubsetsStrategy(SubsetsStrategy):
    def select_subsets(self, values: tp.List[tp.Any]) -> tp.List[tp.List[tp.Any]]:
        return powerset(values, include_empty=False)
