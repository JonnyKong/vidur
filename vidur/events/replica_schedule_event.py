from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id

        self._batches = []

        # Metrics for logging
        self.memory_usage_percent: float = 0.0
        self.request_queue_len: int = 0
        self.running_queue_len: int = 0

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)

        # Get things to log before the scheduler actually runs, otherwise we
        # are undercounting because the batch is already removed from the
        # scheduler queues
        self.request_queue_len = replica_scheduler.num_pending_requests
        if hasattr(replica_scheduler, 'num_running_requests'):
            self.running_queue_len = replica_scheduler.num_running_requests

        self._batches = replica_scheduler.on_schedule()

        if not self._batches:
            return []

        self.memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, self.memory_usage_percent
        )

        for batch in self._batches:
            batch.on_schedule(self.time)

        # Profiled on A40 node
        cpu_overhead_us = max(118.1656 * self.running_queue_len - 80.8321, 0)

        return [
            BatchStageArrivalEvent(
                self.time + cpu_overhead_us / 1e6,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches
        ]

    def to_dict(self):
        if not self._batches:
            # ReplicaScheduleEvent is triggered by both batch completion and
            # request arrival.  Thus, scheduling only occurs for a subset of
            # these events, when the number of outstanding batches is less than
            # the number of pipeline (PP) stages. We log only when scheduling
            # actually takes place.
            return None
        else:
            return {
                "time": self.time,
                "event_type": self.event_type,
                "replica_id": self._replica_id,
                "batch_ids": [batch.id for batch in self._batches],
                "memory_usage_percent": self.memory_usage_percent,
                "request_queue_len": self.request_queue_len,
                "running_queue_len": self.running_queue_len,
            }

    def to_chrome_trace(self) -> dict:
        if not self._batches:
            return None
        else:
            return {
                "name": "Sys metrics",
                "ph": "C",
                "ts": self.time * 1e6,
                "pid": 0,
                "tid": 0,
                "args": {
                    "memory_usage_percent": self.memory_usage_percent,
                    "request_queue_len": self.request_queue_len,
                    "running_queue_len": self.running_queue_len,
                },
            }
