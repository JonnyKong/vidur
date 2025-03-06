from pathlib import Path
from typing import List
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
            --no-metrics_config_write_json_trace 
            --no-metrics_config_save_table_to_wandb 
            --no-metrics_config_store_plots 
            --no-metrics_config_store_operation_metrics 
            --no-metrics_config_store_token_completion_metrics 
            --no-metrics_config_store_request_metrics 
            --no-metrics_config_store_batch_metrics 
            --no-metrics_config_store_utilization_metrics 
            --no-metrics_config_keep_individual_batch_metrics 
        """
        self.step_size_seconds = step_size_seconds

        self.observation_space = gym.spaces.Box(0, 100, shape=(2,))

        self.freq_choices = A40_FREQ_CHOICES
        self.action_space = gym.spaces.Discrete(len(self.freq_choices))

        self.episode_id = -1

        # These will be initialized on every reset()
        self.config: SimulationConfig = SimulationConfig.create_from_args_str(args_str)
        self.simulator: Optional[Simulator] = None
        self.last_step_time: float = 0.0

    def _get_obs(self):
        assert self.simulator
        global_scheduler = self.simulator.scheduler
        replica_scheduler = next(iter(global_scheduler._replica_schedulers.values()))
        states = replica_scheduler.get_states()
        return np.array([
            states['memory_usage_percent'],
            states['waiting_queue_len'],
        ], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.simulator:
            self.simulator._write_output()
            src_log_dir = Path(self.config.metrics_config.output_dir)
            dst_log_dir = src_log_dir.parent / f'episode_{self.episode_id:06d}'
            src_log_dir.rename(dst_log_dir)

        super().reset(seed=seed)
        self.episode_id += 1

        # This will re-create logging dir with a new timestamp
        self.config.metrics_config.__post_init__()

        # Log chrome traces regularly
        self.config.metrics_config.enable_chrome_trace = (self.episode_id % 10 == 0)

        self.simulator = Simulator(self.config)
        # Use highest freq in the beginning
        self.simulator.set_freq(max(self.freq_choices))

        self.last_step_time = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        assert self.simulator

        terminated = False

        freq = self.freq_choices[action]
        self.simulator.set_freq(freq)

        replica_scheduler_states = []
        while self.simulator.get_time() < self.last_step_time + self.step_size_seconds:
            if not self.simulator._event_queue or self.simulator._terminate:
                terminated = True
                break
            s = self.simulator.step()
            if s:
                replica_scheduler_states.append(s)

        self.last_step_time = self.simulator.get_time()

        # terminate if overloads too much, and give a negative reward
        observation = self._get_obs()
        # reward = self.calc_reward(replica_scheduler_states)
        reward = 0.5

        if self.is_overloaded(replica_scheduler_states):
            print('Env terminated because waiting queue grows too long')
            terminated = True

        return observation, reward, terminated, False, self._get_info()

    @staticmethod
    def calc_reward(replica_scheduler_states: List[dict]) -> float:
        if len(replica_scheduler_states) > 0:
            mean_waiting_queue_size = np.mean([s['waiting_queue_len']
                                               for s in replica_scheduler_states])
            return float(1.0 - mean_waiting_queue_size / (mean_waiting_queue_size + 20))
        else:
            return 0.0

    @staticmethod
    def is_overloaded(replica_scheduler_states: List[dict]) -> bool:
        if len(replica_scheduler_states) > 0:
            mean_waiting_queue_size = np.mean([s['waiting_queue_len']
                                               for s in replica_scheduler_states])
            return float(mean_waiting_queue_size) >= 200
        else:
            return False


gym.register(
    id="gymnasium_env/VidurSimulatorEnv",
    entry_point=VidurSimulatorEnv,
)

if __name__ == "__main__":
    env = gym.make("gymnasium_env/VidurSimulatorEnv")
