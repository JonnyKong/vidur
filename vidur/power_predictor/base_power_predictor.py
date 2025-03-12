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
