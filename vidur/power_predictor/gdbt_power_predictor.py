import numpy as np
from lightgbm import Booster

from vidur.config import GdbtPowerPredictorConfig
from vidur.entities import Batch
from vidur.logger import init_logger
from vidur.power_predictor import BasePowerPredictor

logger = init_logger(__name__)


class GdbtPowerPredictor(BasePowerPredictor):
    def __init__(self,
                 power_predictor_config: GdbtPowerPredictorConfig):
        super().__init__(power_predictor_config)

        logger.info(f'Loading GDBT predictor from: {power_predictor_config.model_input_file}')
        self._model = Booster(model_file=power_predictor_config.model_input_file)

    def predict(self, batch: Batch, freq: int) -> float:
        prefill_lens = batch.prefill_lens
        decode_lens = batch.decode_lens

        prefill_batch_size = len(prefill_lens)
        prefill_len_sum = np.sum(prefill_lens) if prefill_batch_size > 0 else 0
        prefill_len_std = np.std(prefill_lens) if prefill_batch_size > 0 else 0.0
        prefill_len_max = np.max(prefill_lens) if prefill_batch_size > 0 else 0

        decode_batch_size = len(decode_lens)
        decode_len_sum = np.sum(decode_lens) if decode_batch_size > 0 else 0
        decode_len_std = np.std(decode_lens) if decode_batch_size > 0 else 0.0
        decode_len_max = np.max(decode_lens) if decode_batch_size > 0 else 0

        # Must match: https://github.com/JonnyKong/vllm/blob/main/benchmarks/power_model.py
        x = np.array([
            freq,
            prefill_batch_size,
            prefill_len_sum,
            prefill_len_std,
            prefill_len_max,
            decode_batch_size,
            decode_len_sum,
            decode_len_std,
            decode_len_max,
        ], dtype=np.float32)
        return self._model.predict([x])[0]
