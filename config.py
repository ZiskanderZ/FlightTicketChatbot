SEED = 23

dataset_path = './dataset.hf'

hf_token_read = "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"
model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_token_write = ''
hub_name = 'zkv/llama-2-7b-chat-hf-assistant-air'

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