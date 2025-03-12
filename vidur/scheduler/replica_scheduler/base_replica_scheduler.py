from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

import numpy as np

from vidur.config import BaseReplicaSchedulerConfig
from vidur.config import BaseRequestGeneratorConfig
from vidur.config import ReplicaConfig
from vidur.entities import Batch
from vidur.entities import Replica
from vidur.entities import Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.power_predictor import BasePowerPredictor
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
        power_predictor: BasePowerPredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(self._replica_config, replica)

        if not self._config.num_blocks:
            self._config.num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }

        self.execution_time_predictor = execution_time_predictor
        self.power_predictor = power_predictor
        self.freq: Optional[int] = None
        self._last_batch_power: float = 0.0
        self._last_batch_idle_duration: float = 0.0
        self._last_batch_time: float = 0.0

    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        return request.num_prefill_tokens

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    def free(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self, time) -> List[Batch]:
        # Record num running requests before actually performing the
        # scheduling, otherwise requests will be taken out of the running queue
        if hasattr(self, 'num_running_requests'):
            running_queue_len = self.num_running_requests
        else:
            running_queue_len = 0

        scheduled_batches: List[Batch] = []
        while self._num_running_batches < self._num_stages:
            batch = self._get_next_batch()
            if not batch:
                break
            scheduled_batches.append(batch)
            self._num_running_batches += 1

        # Only update scheduler states if a batch is actually scheduled
        if len(scheduled_batches) > 0:
            # Power
            if self.execution_time_predictor.freq:
                power_arr = [self.power_predictor.predict(p, self.execution_time_predictor.freq)
                             for p in scheduled_batches]
                self._last_batch_power = float(np.mean(power_arr))
            else:
                print('WARNING: wreq is not yet set, setting predicted power to 0.0')
                self._last_batch_power = 0.0

            # Idle time
            # TODO: Remove this hardcoded number profiled on A40 node
            cpu_overhead_us = max(118.1656 * running_queue_len - 80.8321, 0)
            self._last_batch_idle_duration = cpu_overhead_us / 1e6
            self._last_batch_time = time
        return scheduled_batches

    def set_freq(self, freq: int):
        self.freq = freq
        self.execution_time_predictor.set_freq(freq)

    def set_latency_frequency_predictor_model_path(self, path: str):
        self.execution_time_predictor.set_latency_frequency_predictor_model_path(path)

    def get_states(self, time: float) -> dict:
        # This calculation assumes this function is called at the next
        # scheduling step, before `on_schedule()` is called
        last_batch_duration = time - self._last_batch_time - self._last_batch_idle_duration

        ret = {
            'waiting_queue_len': self.num_pending_requests,
            'memory_usage_percent': self.memory_usage_percent,
            'last_batch_power': self._last_batch_power,
            'last_batch_duration': last_batch_duration,
            'last_batch_idle_duration': self._last_batch_idle_duration,
            'freq': self.execution_time_predictor.freq,
        }
        if hasattr(self, 'num_running_requests'):
            ret['running_queue_len'] = self.num_running_requests
        return ret
