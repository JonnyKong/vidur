import json
import random
import subprocess
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer


def load_sharegpt_ctx_lens(dataset_path: str, filter_by_len: bool) -> Tuple[List[int], List[int]]:
    tokenizer_id = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer_mode = 'auto'  # Default in vllm
    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=tokenizer_mode,
                              trust_remote_code=False)
    num_requests = 16384
    dataset = _sample_sharegpt_requests(dataset_path, num_requests, tokenizer,
                                        fixed_output_len=None, filter_by_len=filter_by_len)
    return (
        [x[1] for x in dataset],
        [x[2] for x in dataset],
    )


# Copied from: https://github.com/vllm-project/vllm/blob/5fe6bf29d657518eb4251981ada9f8c4f34dbbde/benchmarks/benchmark_serving.py#L88
def _sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    filter_by_len: bool = True,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.seed(0)
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int, None]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if filter_by_len:
            if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset


def download_file_if_not_exists(url, output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        print(f"Downloading file from {url} to {output_path}...")
        subprocess.run(["wget", "-O", str(output_path), url], check=True)
    else:
        print('File already exists, skipping download: {output_path}')


if __name__ == '__main__':
    url = 'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json'
    output_path = Path('/tmp') / url.split('/')[-1]

    download_file_if_not_exists(url, output_path)

    num_prefill_tokens, num_decode_tokens = load_sharegpt_ctx_lens(
        str(output_path), filter_by_len=True)
    pd.DataFrame({
        'num_prefill_tokens': num_prefill_tokens,
        'num_decode_tokens': num_decode_tokens,
    }).to_csv('../data/processed_traces/sharegpt_v3_filtered.csv', index=False)
