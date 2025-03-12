from vidur.entities import Batch
from vidur.power_predictor import BasePowerPredictor


class DummyPowerPredictor(BasePowerPredictor):
    def predict(self, batch: Batch, freq: int) -> float:
        return 0.0
