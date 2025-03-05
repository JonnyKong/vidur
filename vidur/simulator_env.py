from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

A40_FREQ_CHOICES = [
    540,
    660,
    780,
    900,
    1020,
    1140,
    1260,
    1380,
    1500,
    1620,
    1740,
]


class VidurSimulatorEnv(gym.Env):
    def __init__(self, step_size_seconds: float = 1.0):
        current_file_path = Path(__file__)
        project_root_path = current_file_path.parent.parent

        args_str = f"""
            --replica_config_device a40 
            --replica_config_model_name meta-llama/Meta-Llama-3-8B 
            --cluster_config_num_replicas 1 
            --replica_config_tensor_parallel_size 1 
            --replica_config_num_pipeline_stages 1 
            --request_generator_config_type synthetic 
            --synthetic_request_generator_config_num_requests 20000 
            --length_generator_config_type trace 
            --trace_request_length_generator_config_max_tokens 16384 
            --trace_request_length_generator_config_trace_file {project_root_path}/data/processed_traces/sharegpt_v3_filtered.csv 
            --interval_generator_config_type gamma 
            --gamma_request_interval_generator_config_qps 10 
            --gamma_request_interval_generator_config_cv 1.414 
            --replica_scheduler_config_type sarathi 
            --sarathi_scheduler_config_batch_size_cap 2048 
            --sarathi_scheduler_config_chunk_size 2048 
            --sarathi_scheduler_config_batch_size_cap 8192 
            --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 
            --random_forrest_execution_time_predictor_config_prediction_max_batch_size 2048 
            --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384
        """
        self.step_size_seconds = step_size_seconds

        self.observation_space = gym.spaces.Dict({
            'memory_usage_percent': gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
        })

        self.freq_choices = A40_FREQ_CHOICES
        self.action_space = gym.spaces.Discrete(len(self.freq_choices))

        self.config: SimulationConfig = SimulationConfig.create_from_args_str(args_str)
        self.simulator = Simulator(self.config)
        self.last_step_time: Optional[float] = None

        # Use highest freq in the beginning
        self.simulator.set_freq(max(self.freq_choices))

    def _get_obs(self):
        global_scheduler = self.simulator.scheduler
        replica_scheduler = global_scheduler.get_replica_scheduler(0)
        states = replica_scheduler.get_states()
        return {
            'memory_usage_percent': states['memory_usage_percent'],
        }

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.simulator = Simulator(self.config)
        self.last_step_time = None

    def step(self, action):
        terminated = False

        freq = self.freq_choices[action]
        self.simulator.set_freq(freq)

        while (
            self.last_step_time is None
            or self.simulator.get_time() < self.last_step_time + self.step_size_seconds
        ):
            if not self.simulator._event_queue or self.simulator._terminate:
                terminated = True
                break
            # TODO: terminate if overloads too much, and give a negative reward
            self.simulator.step()

        self.last_step_time = self.simulator.get_time()

        truncated = False
        reward = 0.0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="gymnasium_env/VidurSimulatorEnv",
    entry_point=VidurSimulatorEnv,
)

if __name__ == "__main__":
    env = gym.make("gymnasium_env/VidurSimulatorEnv")
