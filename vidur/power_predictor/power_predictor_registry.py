from vidur.power_predictor import DummyPowerPredictor
from vidur.power_predictor import GdbtPowerPredictor
from vidur.types import PowerPredictorType
from vidur.utils.base_registry import BaseRegistry


class PowerPredictorRegistry(BaseRegistry):
    pass


PowerPredictorRegistry.register(
    PowerPredictorType.DUMMY, DummyPowerPredictor,
)
PowerPredictorRegistry.register(
    PowerPredictorType.GDBT, GdbtPowerPredictor,
)
