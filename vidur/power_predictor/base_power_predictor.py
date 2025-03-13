from abc import ABC
from abc import abstractmethod
from typing import Optional

from vidur.config import BasePowerPredictorConfig
from vidur.entities import Batch


class BasePowerPredictor(ABC):
    def __init__(self,
                 power_predictor_config: BasePowerPredictorConfig):
        self._config = power_predictor_config

        self.freq: Optional[int] = None

    @abstractmethod
    def predict(self, batch: Batch, freq: int) -> float:
        pass

    def predict_idle_power(self) -> float:
        # TODO: remove hardcoded tdp
        a40_tdp = 300.0
        return 0.25 * a40_tdp
