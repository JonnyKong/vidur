from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, ExecutionTime
from vidur.logger import init_logger

logger = init_logger(__name__)


class BaseExecutionTimePredictor(ABC):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config

        # get configs
        self._replica_scheduler_provider = str(replica_scheduler_config.get_type())
        self._block_size = replica_scheduler_config.block_size
        self._cache_dir = metrics_config.cache_dir
        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )

        self.freq: Optional[int] = None

        self.latency_frequency_predictor_model_path = None

        self._latency_freq_model_prefill = None
        self._latency_freq_model_decode = None
        self._latency_freq_model_hybrid = None
        


    def get_execution_time(self, batch: Batch, pipeline_stage: int) -> ExecutionTime:
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._get_pipeline_parallel_communication_time(batch)
            )

        if self._replica_config.tensor_parallel_size == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)
            )

        latency_from_freq_model = 0
        if self.latency_frequency_predictor_model_path is not None:
            freq = self.freq
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

            if (prefill_batch_size > 0) and (decode_batch_size == 0):
                model_input_prefill = np.array([freq, prefill_batch_size, prefill_len_sum, prefill_len_max, prefill_len_std]).reshape(1, -1)
                latency_from_freq_model = self._latency_freq_model_prefill.predict(model_input_prefill)
                
            elif (prefill_batch_size == 0) and (decode_batch_size > 0):
                model_input_decode = np.array([freq, decode_batch_size, decode_len_sum, decode_len_max, decode_len_std]).reshape(1, -1)
                latency_from_freq_model = self._latency_freq_model_decode.predict(model_input_decode)
            else:
                model_input_hybrid = np.array([freq, decode_batch_size, prefill_batch_size, decode_len_sum, prefill_len_sum, decode_len_max, prefill_len_max, decode_len_std, prefill_len_std]).reshape(1, -1)
                latency_from_freq_model = self._latency_freq_model_hybrid.predict(model_input_hybrid)
            latency_from_freq_model = latency_from_freq_model.item()
            t = ExecutionTime(
                self._num_layers_per_pipeline_stage,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                tensor_parallel_communication_time,
                pipeline_parallel_communication_time,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                latency_from_freq_model,
            )
        else:
            t = ExecutionTime(
                self._num_layers_per_pipeline_stage,
                self._get_attention_rope_execution_time(batch),
                self._get_attention_kv_cache_save_execution_time(batch),
                self._get_attention_decode_execution_time(batch),
                self._get_attention_prefill_execution_time(batch),
                self._get_attention_layer_pre_proj_execution_time(batch),
                self._get_attention_layer_post_proj_execution_time(batch),
                self._get_mlp_layer_up_proj_execution_time(batch),
                self._get_mlp_layer_down_proj_execution_time(batch),
                self._get_mlp_layer_act_execution_time(batch),
                self._get_attn_norm_layer_act_execution_time(batch),
                self._get_mlp_norm_layer_act_execution_time(batch),
                self._get_add_layer_act_execution_time(batch),
                tensor_parallel_communication_time,
                pipeline_parallel_communication_time,
                self._get_schedule_time(batch),
                self._get_sampler_e2e_time(batch),
                self._get_prepare_inputs_e2e_time(batch),
                self._get_process_model_outputs_time(batch),
                self._get_ray_comm_time(batch),
                latency_from_freq_model,
            )
            if self.freq is not None:
                self.scale_execution_time_by_freq(t, self.freq)
        return t

    @abstractmethod
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_schedule_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_ray_comm_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    def set_freq(self, freq):
        self.freq = freq

    def set_latency_frequency_predictor_model_path(self, path: str):
        self.latency_frequency_predictor_model_path = path
        if self.latency_frequency_predictor_model_path:
            try:
                with open(self.latency_frequency_predictor_model_path + "/batch_latency_predictor_A40-LLama3-8B_prefill-only.pkl", 'rb') as f:
                    self._latency_freq_model_prefill = pickle.load(f)
                    print("Loaded prefill model")
                with open(self.latency_frequency_predictor_model_path + "/batch_latency_predictor_A40-LLama3-8B_decode-only.pkl", 'rb') as f:
                    self._latency_freq_model_decode = pickle.load(f)
                    print("Loaded decode model")
                with open(self.latency_frequency_predictor_model_path + "/batch_latency_predictor_A40-LLama3-8B_hybrid.pkl", 'rb') as f:
                    self._latency_freq_model_hybrid = pickle.load(f)
                    print("Loaded hybrid model")
            except FileNotFoundError:
                self._config.latency_frequency_predictor_enabled = None
                logger.error(f"Latency frequency model not found at {self.latency_frequency_predictor_model_path}")

    @staticmethod
    def scale_execution_time_by_freq(t: ExecutionTime, freq: int) -> None:
        factor = {
            210: 5.0,
            360: 3.5,
            510: 2.5,
            675: 2.0,
            825: 1.50,
            975: 1.30,
            1125: 1.15,
            1275: 1.08,
            1440: 1.05,
            1590: 1.02,
            1740: 1.00,
        }[freq]
        BaseExecutionTimePredictor.scale_execution_time_by_factor(t, factor)

    @staticmethod
    def scale_execution_time_by_factor(t: ExecutionTime, factor: float) -> None:
        t._attention_rope_execution_time *= factor
        t._attention_kv_cache_save_execution_time *= factor
        t._attention_decode_execution_time *= factor
        t._attention_prefill_execution_time *= factor
        t._attention_layer_pre_proj_execution_time *= factor
        t._attention_layer_post_proj_execution_time *= factor
        t._mlp_layer_up_proj_execution_time *= factor
        t._mlp_layer_down_proj_execution_time *= factor
        t._mlp_layer_act_execution_time *= factor
        t._attn_norm_time *= factor
        t._mlp_norm_time *= factor
        t._add_time *= factor
        t._tensor_parallel_communication_time *= factor
        t._pipeline_parallel_communication_time *= factor
        t._schedule_time *= factor
        t._sampler_e2e_time *= factor
        t._prepare_inputs_e2e_time *= factor
        t._process_model_outputs_time *= factor
        t._ray_comm_time *= factor
