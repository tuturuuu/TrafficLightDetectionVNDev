from ..models.lts import parameter_free_map
from ..registry import SELECTORS
from .relevance import RelevanceSelector


@SELECTORS.register('param_free_avg')
class ParameterFreeSelector(RelevanceSelector):
    def predict_map(self, features):
        return parameter_free_map(features)
