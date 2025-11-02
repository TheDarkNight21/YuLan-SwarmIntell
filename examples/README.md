# SwarmBench Examples - Local Model Support

This directory contains example scripts demonstrating how to use local HuggingFace models with SwarmBench.

## Quick Start

### Using Local Models

```python
from swarmbench import SwarmFramework

# Create local model configuration
model_cfg = SwarmFramework.local_model_config(
    'meta-llama/Llama-3.2-3B-Instruct',  # HuggingFace model ID
    device="auto",                        # Use GPU if available
    temperature=0.7,
    use_vllm=True                         # Fast inference with vLLM
)

# Submit a task
SwarmFramework.submit(
    'my_experiment',
    model_cfg,
    'Transport',
    log_dir='./logs',
    num_agents=10,
    max_round=100
)

# Run all tasks
SwarmFramework.run_all()
```

### Using API Models (Original Method)

```python
# API-based model configuration (unchanged from original)
model_cfg = SwarmFramework.model_config(
    'gpt-4o-mini',
    'YOUR_API_KEY',
    'https://openrouter.ai/api/v1'
)
```

## Example Scripts

### `eval_local.py`
Complete example using only local models. Demonstrates:
- Small models (3B parameters) for fast testing
- Larger models (7B+ parameters) for better performance
- CPU inference (slow, but works without GPU)

**Run it:**
```bash
python examples/eval_local.py
```

### `eval_mixed.py`
Advanced example mixing API and local models. Useful for:
- Comparing local vs API model performance
- Using different models for different tasks
- Cost optimization (free local inference + paid API for critical tasks)

**Run it:**
```bash
python examples/eval_mixed.py
```

## Local Model Configuration Options

### `local_model_config()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | **required** | HuggingFace model ID (e.g., `'meta-llama/Llama-3.2-3B-Instruct'`) or local path |
| `device` | str | `"auto"` | Device to use: `"auto"` (GPU if available), `"cuda"`, or `"cpu"` |
| `max_new_tokens` | int | `512` | Maximum tokens to generate per response |
| `temperature` | float | `0.7` | Sampling temperature (0.0 = greedy, 1.0+ = creative) |
| `use_vllm` | bool | `True` | Use vLLM for faster inference (GPU only, falls back to Transformers) |

## Recommended Models

### Small Models (Fast, <10GB VRAM)
- **Llama-3.2-3B-Instruct** - `meta-llama/Llama-3.2-3B-Instruct`
  - Excellent for testing and rapid iteration
  - ~6GB VRAM required
  - Good instruction following

- **Phi-3-mini-4k-instruct** - `microsoft/Phi-3-mini-4k-instruct`
  - Even smaller (3.8B parameters)
  - Works on CPU (though slow)
  - Good for debugging

### Medium Models (Better Quality, 14-20GB VRAM)
- **Llama-3.1-8B-Instruct** - `meta-llama/Llama-3.1-8B-Instruct`
  - Strong reasoning capabilities
  - ~14GB VRAM required
  - Great balance of speed and quality

- **Mistral-7B-Instruct-v0.3** - `mistralai/Mistral-7B-Instruct-v0.3`
  - Fast and efficient
  - Good instruction following
  - ~14GB VRAM required

### Large Models (Best Quality, 40GB+ VRAM)
- **Llama-3.1-70B-Instruct** - `meta-llama/Llama-3.1-70B-Instruct`
  - Excellent reasoning and planning
  - Requires 2+ high-end GPUs or quantization
  - Consider using via API instead

## Hardware Requirements

### GPU Inference (Recommended)

| Model Size | VRAM Required | Recommended GPU |
|------------|---------------|-----------------|
| 3B params | 6-8GB | RTX 3060, RTX 4060 |
| 7-8B params | 14-16GB | RTX 3090, RTX 4080, A5000 |
| 13B params | 26-30GB | A6000, RTX 6000 Ada |
| 70B params | 140GB+ | 2x A100 or quantized versions |

### CPU Inference
- **Possible but VERY slow** (10-100x slower than GPU)
- Use small models only (<3B parameters)
- Reduce `num_agents` and `max_round` significantly
- Set `use_vllm=False` (vLLM requires GPU)

## Performance Tips

### 1. Model Caching
Models are automatically cached after first download:
```
~/.cache/huggingface/hub/
```
Subsequent runs will be much faster (no download).

### 2. Memory Management
- Run one task at a time for large models: `run_all(max_parallel=1)`
- Multiple agents share the same model in memory (efficient!)
- Close other GPU applications before running

### 3. Speed Optimization
- **Use vLLM on GPU** (2-10x faster than Transformers)
- **Reduce `max_new_tokens`** if responses are too long
- **Lower `temperature`** for faster sampling (0.5-0.7 is good)
- **Use smaller models** for experimentation, scale up for final evaluation

### 4. Quality Optimization
- **Increase `temperature`** for more creative/diverse responses (0.7-0.9)
- **Use larger models** (7B+ for complex reasoning tasks)
- **Adjust `max_new_tokens`** based on task complexity

## Troubleshooting

### Out of Memory (OOM) Errors
```
CUDA out of memory
```
**Solutions:**
1. Use a smaller model
2. Set `max_parallel=1` in `run_all()`
3. Reduce `num_agents`
4. Use CPU: `device="cpu"` (will be slow)

### Model Download Failures
```
Connection error / Timeout
```
**Solutions:**
1. Check internet connection
2. Try again (HuggingFace can be slow)
3. Use HuggingFace CLI to pre-download:
   ```bash
   huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
   ```

### vLLM Import Errors
```
ModuleNotFoundError: No module named 'vllm'
```
**Solution:** vLLM is optional. Set `use_vllm=False` to use Transformers backend:
```python
SwarmFramework.local_model_config(
    'meta-llama/Llama-3.2-3B-Instruct',
    use_vllm=False  # Use Transformers instead
)
```

### Slow CPU Inference
**This is expected!** CPU inference is 10-100x slower than GPU.
- Use small models (<3B params)
- Reduce `num_agents` (try 2-3)
- Reduce `max_round` (try 5-10)
- Consider using API models instead

## Comparing API vs Local Models

### API Models
✅ **Pros:**
- No hardware requirements
- Latest models (GPT-4, Claude, etc.)
- No setup required
- Consistent performance

❌ **Cons:**
- Costs money per request
- Requires internet connection
- Rate limits
- Less control over inference

### Local Models
✅ **Pros:**
- Free inference after download
- Complete privacy (runs on your machine)
- No rate limits
- Full control over parameters
- Works offline (after download)

❌ **Cons:**
- Requires GPU (or slow on CPU)
- Large disk space for models
- One-time download time
- Smaller models may have lower quality

## Advanced Usage

### Custom Model Paths
Use local model directories instead of HuggingFace IDs:
```python
SwarmFramework.local_model_config(
    '/path/to/my/fine-tuned/model',
    device="cuda"
)
```

### Temperature Sweep
Compare different sampling strategies:
```python
for temp in [0.3, 0.7, 1.0]:
    SwarmFramework.submit(
        f'temp_{temp}',
        SwarmFramework.local_model_config(
            'meta-llama/Llama-3.2-3B-Instruct',
            temperature=temp
        ),
        'Transport',
        seed=42
    )
SwarmFramework.run_all()
```

### Mixed Agent Teams
Different agents can use different models (coming soon):
```python
# Not yet implemented - future feature
models = [
    SwarmFramework.local_model_config('model-A'),  # Agents 0-4
    SwarmFramework.local_model_config('model-B'),  # Agents 5-9
]
SwarmFramework.submit('mixed_team', models, 'Transport')
```

## Further Reading

- [HuggingFace Model Hub](https://huggingface.co/models)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [SwarmBench Paper](https://github.com/YuLan-Team/SwarmBench) (add link)

## Support

If you encounter issues with local model support:
1. Check this README first
2. Open an issue on GitHub with:
   - Your hardware (GPU model, VRAM)
   - Model you're trying to use
   - Full error message
   - Output of `nvidia-smi` (if using GPU)
