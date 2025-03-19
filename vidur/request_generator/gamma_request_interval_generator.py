from scipy.stats import gamma

from vidur.config import GammaRequestIntervalGeneratorConfig
from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class GammaRequestIntervalGenerator(BaseRequestIntervalGenerator):

    def __init__(self, config: GammaRequestIntervalGeneratorConfig):
        super().__init__(config)

        cv = self.config.cv
        self.qps = self.config.qps
        self.gamma_shape = 1.0 / (cv**2)

        # Generate a large set of arrival times in batch on initialization
        gamma_scale = 1.0 / (self.qps * self.gamma_shape)
        self.precomputed_arrival_times = gamma.rvs(
                self.gamma_shape, scale=gamma_scale, size=65536)
        self.index = 0

    def get_next_inter_request_time(self) -> float:
        self.index = (self.index + 1) % len(self.precomputed_arrival_times)
        return self.precomputed_arrival_times[self.index]
