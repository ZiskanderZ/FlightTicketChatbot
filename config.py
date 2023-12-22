import os

OPENAI_API_KEY = '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'
LLAMA_API = 'LL-V7R9KNkGpxNYtGLbuj4yB7zDXVSFx3zpYcklmRQTezz6EB6lVSiq2F6hkB7fSGuT'

SEED = 23

dataset_path = './dataset.hf'

hf_token_read = "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"
model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_token_write = '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'
hub_name = 'zkv/llama-2-7b-chat-hf-air-assistant'

num_test_samples = 128

lora_target_modules = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj",
]

path_to_peft_model = 'peft_model'
path_to_ft_model = 'ft_model'