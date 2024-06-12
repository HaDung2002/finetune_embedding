from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import os
import torch

def get_llm(model_name,
            token,
            temperature=0.7,
            max_new_tokens=256,
            context_window=3900):
    os.environ["HF_TOKEN"] = token
    quantization_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        load_in_4bit=True
    )
    
    llm = HuggingFaceLLM(
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs={"temperature": temperature, "do_sample": True},
        model_name=model_name,
        tokenizer_name=model_name,
        device_map="cuda",
        tokenizer_kwargs={"max_length": context_window},
        model_kwargs={"quantization_config": quantization_config, "torch_dtype": torch.bfloat16},
    )
    return llm