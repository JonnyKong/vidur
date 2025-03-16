from typing import Any

from vidur.execution_time_predictor.linear_regression_execution_time_predictor import (
    LinearRegressionExecutionTimePredictor,
)
from vidur.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from vidur.logger import init_logger
from vidur.types import BaseIntEnum
from vidur.types import ExecutionTimePredictorType
from vidur.utils.base_registry import BaseRegistry

logger = init_logger(__name__)


class ExecutionTimePredictorRegistry(BaseRegistry):
    cached_predictor = None

    @classmethod
    def get_key_from_str(cls, key_str: str) -> ExecutionTimePredictorType:
        return ExecutionTimePredictorType.from_str(key_str)

    @classmethod
    def get(cls, key: BaseIntEnum, *args, **kwargs) -> Any:
        if cls.cached_predictor is None:
            logger.info('Loading ExecutionTimePredictor from disk ...')
            cls.cached_predictor = super().get(key, *args, **kwargs)
        else:
            logger.info('Using cached ExecutionTimePredictor')
        return cls.cached_predictor


ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.RANDOM_FORREST, RandomForrestExecutionTimePredictor
)
ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.LINEAR_REGRESSION, LinearRegressionExecutionTimePredictor
)
