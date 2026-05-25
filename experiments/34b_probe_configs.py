"""Probe modern open models for HQMQ compatibility via config.json."""
import json

from huggingface_hub import hf_hub_download

candidates = [
    "google/gemma-2-9b",
    "google/gemma-2-2b",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "NousResearch/Meta-Llama-3.1-8B",
    "01-ai/Yi-1.5-9B",
    "google/gemma-3-12b-pt",
    "google/gemma-3-4b-pt",
]

for name in candidates:
    try:
        path = hf_hub_download(repo_id=name, filename="config.json")
        with open(path) as f:
            cfg = json.load(f)
        h = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size")
        na = cfg.get("num_attention_heads") or cfg.get("text_config", {}).get("num_attention_heads")
        nkv = cfg.get("num_key_value_heads") or cfg.get("text_config", {}).get("num_key_value_heads", na)
        nl = cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers")
        head_dim = cfg.get("head_dim") or cfg.get("text_config", {}).get("head_dim")
        d_head = head_dim if head_dim else (h // na if (h and na) else None)
        ok = d_head and d_head % 4 == 0
        arch = cfg.get("model_type", "?")
        # Estimate KV cache size at 32k context, bf16, GQA
        if nl and nkv and d_head:
            kv_mem_gb = 2 * nl * nkv * d_head * 2 * 32768 / 1e9  # K and V, fp16
            kv_mem_str = f"KV@32k={kv_mem_gb:.2f}GB"
        else:
            kv_mem_str = ""
        # Estimate model size in bf16
        if h and nl:
            model_mem_gb_est = 12 * h * h * nl / 1e9  # rough: 12 * h^2 * L
        else:
            model_mem_gb_est = None
        print(f"{name:42s} {arch:14s} L={nl} n_attn={na} n_kv={nkv} d_head={d_head} HQMQ-ok={ok} {kv_mem_str}")
    except Exception as e:
        print(f"{name:42s} ERROR: {str(e)[:100]}")
