from typing import Dict
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
        self.scheduler_states: Dict = {}

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)

        # Record states before the scheduler actually runs, since otherwise we
        # are undercounting because the batch is already removed from the
        # scheduler queues. Also it's because states should correspond to the
        # last finished batch, not the newly scheduled, in-execution batch
        self.scheduler_states = replica_scheduler.get_states(self.time)

        self._batches = replica_scheduler.on_schedule(self.time)

        if not self._batches:
            return []

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent
        )

        for batch in self._batches:
            batch.on_schedule(self.time)

        return [
            BatchStageArrivalEvent(
                self.time + replica_scheduler._last_batch_idle_duration,
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
                **self.scheduler_states,
            }

    def to_chrome_trace(self) -> dict:
        if not self._batches:
            return None
        else:
            return {
                "name": "stats",
                "ph": "C",
                "ts": self.time * 1e6,
                "pid": 0,
                "tid": 0,
                "args": {
                    **self.scheduler_states,
                },
            }
