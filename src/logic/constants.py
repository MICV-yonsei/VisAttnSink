import torch

MODEL_LLM = {
    "llava-v1.5-7b": "llama-v2-7b",
    "llava-v1.5-13b": "llama-v2-13b",
    "llava-v1.6-vicuna-13b": "llama-v2-13b",
}
DIM_SINK = {
    "llama-v2-7b": torch.tensor([2533, 1415]),
    "llama-v2-13b": torch.tensor([2100, 4743]),
}
