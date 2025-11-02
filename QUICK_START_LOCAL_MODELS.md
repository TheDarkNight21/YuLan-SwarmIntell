# Quick Start: Local Models in SwarmBench

## 5-Minute Setup

### 1. Verify Your Environment
```bash
# Check if you have GPU
#python -c "import torch; print('GPU available:', torch.cuda.is_available())"

python -c "import torch; print('CUDA version:', torch.version.cuda); print('Torch version:', torch.__version__); print('cuDNN available:', torch.backends.cudnn.is_available())"

# Check VRAM (optional, for Linux/CUDA)
nvidia-smi
```

### 2. Run Your First Local Model
```python
from swarmbench import SwarmFramework

# Configure local model (auto-downloads from HuggingFace)
SwarmFramework.submit(
    'my_first_local_run',
    SwarmFramework.local_model_config(
        'meta-llama/Llama-3.2-3B-Instruct'  # 3B model, ~6GB VRAM
    ),
    'Transport',      # Task
    log_dir='./logs',
    num_agents=5,     # Start small
    max_round=20      # Short run for testing
)

SwarmFramework.run_all()
```

### 3. Run the Test
```bash
python test_local_model.py
```

## What Just Happened?

1. **Model Download**: First run downloads ~6GB from HuggingFace (one-time)
2. **Model Loading**: Model loads into GPU memory (~6GB VRAM)
3. **Inference**: 5 agents make decisions using the local model
4. **Results**: Logs saved to `./logs/`

## Next Steps

### Want Faster Inference?
Use vLLM (already installed, auto-enabled on GPU):
```python
SwarmFramework.local_model_config(
    'meta-llama/Llama-3.2-3B-Instruct',
    use_vllm=True  # 2-10x faster (default)
)
```

### Want Better Quality?
Use a larger model (requires more VRAM):
```python
SwarmFramework.local_model_config(
    'meta-llama/Llama-3.1-8B-Instruct',  # Needs ~14GB VRAM
    temperature=0.7  # Higher = more creative
)
```

### No GPU?
Use CPU (much slower, not recommended for full runs):
```python
SwarmFramework.local_model_config(
    'microsoft/Phi-3-mini-4k-instruct',  # Smaller model
    device="cpu",
    use_vllm=False,  # vLLM requires GPU
    max_new_tokens=256
)
```

### Mix API and Local Models?
```python
# API model (e.g., GPT-4)
SwarmFramework.submit('api',
    SwarmFramework.model_config('gpt-4o-mini', 'KEY', 'URL'),
    'Transport')

# Local model
SwarmFramework.submit('local',
    SwarmFramework.local_model_config('meta-llama/Llama-3.2-3B-Instruct'),
    'Transport')

SwarmFramework.run_all(max_parallel=2)  # Run both
```

## Common Issues

### "Out of memory"
→ Use smaller model or fewer parallel tasks:
```python
SwarmFramework.run_all(max_parallel=1)  # One at a time
```

### "Download is slow"
→ Normal for first run (5-15 min). Models cached at `~/.cache/huggingface/`

### "ImportError: vllm"
→ vLLM is optional. Disable it:
```python
SwarmFramework.local_model_config(model, use_vllm=False)
```

## Full Documentation

- **Examples**: `examples/eval_local.py`, `examples/eval_mixed.py`
- **Guide**: `examples/README.md`
- **Implementation Details**: `LOCAL_MODEL_IMPLEMENTATION.md`
- **Test Suite**: `test_local_model.py`

## Recommended Models

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| Llama-3.2-3B-Instruct | 3B | 6GB | Testing, fast iteration |
| Llama-3.1-8B-Instruct | 8B | 14GB | Production, good quality |
| Phi-3-mini-4k-instruct | 3.8B | 6GB | CPU-friendly |

## Help

Something not working? Run the test suite:
```bash
python test_local_model.py
```

Check the troubleshooting section in `examples/README.md`.

---

**Happy experimenting with local models! 🚀**
