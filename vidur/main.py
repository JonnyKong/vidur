from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.set_freq(1740)
    if config.latency_frequency_predictor_model_path:
        simulator.set_latency_frequency_predictor_model_path(
            config.latency_frequency_predictor_model_path)
    simulator.run()


if __name__ == "__main__":
    main()
