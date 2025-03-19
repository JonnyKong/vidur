import shutil
from pathlib import Path
from typing import List
from typing import Optional

import gymnasium as gym
import numpy as np

from vidur.config import SimulationConfig
from vidur.simulator import Simulator

A40_FREQ_CHOICES = [210, 360, 510, 675, 825, 975, 1125, 1275, 1440, 1590, 1740]
A40_TDP = 300


class VidurSimulatorEnv(gym.Env):
    def __init__(self, env_idx: int = 0, step_size_seconds: float = 1.0,
                 log_dir: str='./simulator_outputs', extra_vidur_args_str=''):
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
            --no-metrics_config_enable_chrome_trace 
            --power_predictor_config_type gdbt 
            --gdbt_power_predictor_config_model_input_file {Path(__file__).parent.parent}/artifacts/power_model/a40_llama8-3b/power_model.txt 
            --latency_frequency_predictor_model_path {Path(__file__).parent.parent}/artifacts/latency_model/a40_llama8-3b 
            --metrics_config_output_dir_root {log_dir} 
        """
        self.env_idx = env_idx
        self.step_size_seconds = step_size_seconds

        self.observation_space = gym.spaces.Box(0, 100, shape=(2,))

        self.freq_choices = A40_FREQ_CHOICES
        self.action_space = gym.spaces.Discrete(len(self.freq_choices))

        self.episode_id = -1

        # These will be initialized on every reset()
        self.config: SimulationConfig = SimulationConfig.create_from_args_str(
                args_str + ' ' + extra_vidur_args_str)
        self.simulator: Optional[Simulator] = None
        self.last_step_time: float = 0.0

    @staticmethod
    def _get_obs(replica_scheduler_states: List[dict]):
        if len(replica_scheduler_states) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        busy_energy = np.sum([s['last_batch_busy_power'] * s['last_batch_busy_duration']
                              for s in replica_scheduler_states])
        idle_energy = np.sum([s['last_batch_idle_power'] * s['last_batch_idle_duration']
                              for s in replica_scheduler_states])
        total_duration = np.sum([s['last_batch_busy_duration'] + s['last_batch_idle_duration']
                                for s in replica_scheduler_states])
        avg_power = (busy_energy + idle_energy) / total_duration
        avg_power_util = avg_power / A40_TDP * 100

        avg_waiting_queue_len = np.mean([s['waiting_queue_len']
                                        for s in replica_scheduler_states])
        avg_memory_usage_percent = np.mean([s['memory_usage_percent']
                                        for s in replica_scheduler_states])

        return np.array([
            avg_memory_usage_percent,
            avg_waiting_queue_len,
        ], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.simulator:
            if self.config.metrics_config.enable_chrome_trace:
                self.simulator._write_output()
                src_log_dir = Path(self.config.metrics_config.output_dir)
                dst_log_dir = src_log_dir.parent / f'episode_{self.episode_id:06d}'
                src_log_dir.rename(dst_log_dir)
            else:
                shutil.rmtree(self.config.metrics_config.output_dir, ignore_errors=True)

        super().reset(seed=seed)
        self.episode_id += 1

        # Log chrome traces regularly
        self.config.metrics_config.enable_chrome_trace = (
            self.env_idx == 0 and self.episode_id % 50 == 0)

        # Use highest freq in the beginning
        self.simulator = Simulator(self.config)
        self.simulator.set_freq(max(self.freq_choices))

        self.last_step_time = 0.0

        observation = self._get_obs(replica_scheduler_states=[])
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
        observation = self._get_obs(replica_scheduler_states)
        reward = self._calc_reward(replica_scheduler_states)
        # reward = 0.5

        if self.is_overloaded(replica_scheduler_states):
            print('Env terminated because waiting queue grows too long')
            terminated = True
            reward = -1.0

        return observation, reward, terminated, False, self._get_info()

    @staticmethod
    def _calc_reward(replica_scheduler_states: List[dict]) -> float:
        if len(replica_scheduler_states) > 0:
            mean_waiting_queue_size = np.mean([s['waiting_queue_len']
                                               for s in replica_scheduler_states])
            busy_energy = np.sum([s['last_batch_busy_power'] * s['last_batch_busy_duration']
                                  for s in replica_scheduler_states])
            idle_energy = np.sum([s['last_batch_idle_power'] * s['last_batch_idle_duration']
                                  for s in replica_scheduler_states])
            total_duration = np.sum([s['last_batch_busy_duration'] + s['last_batch_idle_duration']
                                    for s in replica_scheduler_states])
            tbt_p99 = np.percentile([s['last_batch_busy_duration'] + s['last_batch_idle_duration']
                                    for s in replica_scheduler_states], 99)
            avg_power = (busy_energy + idle_energy) / total_duration
            tbt = np.sum([s['last_batch_busy_duration'] + s['last_batch_idle_duration']
                          for s in replica_scheduler_states])

            reward = (1 - (avg_power / A40_TDP)) - (0.01 * mean_waiting_queue_size)
            return reward
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
