#!/bin/bash

set -e

python -m vidur.main \
    --replica_config_device a40 \
    --replica_config_model_name meta-llama/Meta-Llama-3-8B \
    --cluster_config_num_replicas 1 \
    --replica_config_tensor_parallel_size 1 \
    --replica_config_num_pipeline_stages 1 \
    --request_generator_config_type synthetic \
    --synthetic_request_generator_config_num_requests 20000 \
    --length_generator_config_type trace \
    --trace_request_length_generator_config_max_tokens 16384 \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/sharegpt_v3_filtered.csv \
    --interval_generator_config_type gamma \
    --gamma_request_interval_generator_config_qps 7 \
    --gamma_request_interval_generator_config_cv 1.414 \
    --replica_scheduler_config_type sarathi \
    --sarathi_scheduler_config_batch_size_cap 2048 \
    --sarathi_scheduler_config_chunk_size 2048 \
    --sarathi_scheduler_config_batch_size_cap 8192 \
    --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
    --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
    --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384
