# Local Model Implementation Summary

## Overview
This document summarizes the implementation of local HuggingFace model support for SwarmBench. Users can now run benchmarks using models on their own hardware instead of relying solely on API services.

## Implementation Date
2025-11-02

## What Was Implemented

### 1. Core Framework Changes

#### `framework/framework.py`
**Extended `ModelConfig` class:**
- Added `model_type` field: `"api"` or `"local"`
- Added local model fields: `model_path`, `device`, `max_new_tokens`, `temperature`, `use_vllm`
- Updated `as_dict()` method to handle both configuration types

**Updated `Framework.start()` method:**
- Added logic to create `Chat` for API models or `LocalChat` for local models
- Maintains backward compatibility with existing API-based workflows

#### `framework/llm.py`
**Created `LocalChat` class:**
- Drop-in replacement for `Chat` with identical interface
- Supports two inference backends:
  - **vLLM**: Fast GPU inference (2-10x faster than Transformers)
  - **Transformers**: Universal fallback, works on CPU and GPU
- Features:
  - Automatic model caching (models shared across agents)
  - Async wrapper for synchronous inference
  - Chat template support (model-specific formatting)
  - Token usage tracking
  - Memory management (conversation history)
  - Auto device selection (GPU if available, else CPU)

**Key Features:**
- Global model cache prevents reloading same model multiple times
- Graceful fallback from vLLM to Transformers if vLLM fails
- Compatible with HuggingFace model IDs and local paths
- Thread-safe async generation using `asyncio.to_thread()`

#### `swarmbench/framework.py`
**Added `local_model_config()` class method:**
- User-friendly API for creating local model configurations
- Parameters: `model_path`, `device`, `max_new_tokens`, `temperature`, `use_vllm`
- Comprehensive docstring with usage examples

**Updated `model_config()` method:**
- Added `model_type = "api"` to maintain backward compatibility
- Added docstring for clarity

### 2. Example Scripts

#### `examples/eval_local.py`
Complete example demonstrating:
- Small model usage (3B parameters)
- Large model usage (7B+ parameters)
- CPU inference (for machines without GPU)
- Commented examples with explanations

#### `examples/eval_mixed.py`
Advanced example showing:
- Mixing API and local models in same run
- Comparing different model types on same task
- Use cases for hybrid approaches

#### `examples/README.md`
Comprehensive documentation covering:
- Quick start guide
- Configuration options reference
- Recommended models by size/quality
- Hardware requirements table
- Performance optimization tips
- Troubleshooting guide
- API vs Local comparison
- Advanced usage patterns

### 3. Testing

#### `test_local_model.py`
Test suite including:
- Configuration creation tests (fast, no model loading)
- Backward compatibility tests for API configs
- Full integration test with actual model loading
- Interactive prompts to avoid accidental long runs
- Detailed error messages and troubleshooting hints

## Files Modified

1. `framework/framework.py` - Extended ModelConfig, updated Framework.start()
2. `framework/llm.py` - Added LocalChat class and model caching
3. `swarmbench/framework.py` - Added local_model_config() method

## Files Created

1. `examples/eval_local.py` - Local model usage examples
2. `examples/eval_mixed.py` - Mixed API/local usage
3. `examples/README.md` - Comprehensive documentation
4. `test_local_model.py` - Test suite
5. `LOCAL_MODEL_IMPLEMENTATION.md` - This document

## Key Design Decisions

### 1. Separate Classes (Chat vs LocalChat)
- **Rationale**: Clean separation of concerns, different initialization needs
- **Benefit**: Easier to maintain, test, and optimize independently

### 2. Global Model Cache
- **Rationale**: Avoid loading same model 10 times (once per agent)
- **Benefit**: Memory efficient, faster startup after first agent

### 3. vLLM with Transformers Fallback
- **Rationale**: vLLM is faster but not universally compatible
- **Benefit**: Best performance when possible, always works

### 4. Backward Compatibility
- **Rationale**: Don't break existing API-based workflows
- **Benefit**: Users can adopt local models gradually

### 5. Async Wrapper for Sync Inference
- **Rationale**: Model inference is synchronous but framework is async
- **Benefit**: Non-blocking inference using thread pool

## Usage Examples

### Basic Local Model
```python
from swarmbench import SwarmFramework

SwarmFramework.submit(
    'my_experiment',
    SwarmFramework.local_model_config(
        'meta-llama/Llama-3.2-3B-Instruct',
        device="auto",
        temperature=0.7
    ),
    'Transport',
    log_dir='./logs'
)

SwarmFramework.run_all()
```

### Backward Compatible API Usage
```python
# This still works exactly as before
SwarmFramework.submit(
    'api_experiment',
    SwarmFramework.model_config(
        'gpt-4o-mini',
        'YOUR_API_KEY',
        'https://openrouter.ai/api/v1'
    ),
    'Transport',
    log_dir='./logs'
)

SwarmFramework.run_all()
```

## Testing Instructions

### Quick Configuration Test (No GPU Needed)
```bash
python -c "
from swarmbench import SwarmFramework
cfg = SwarmFramework.local_model_config('test-model')
assert cfg.model_type == 'local'
print('✓ Local config works')
"
```

### Full Integration Test (Requires GPU)
```bash
python test_local_model.py
```

### Run Examples
```bash
# Local models only
python examples/eval_local.py

# Mix API and local models
python examples/eval_mixed.py
```

## Hardware Requirements

| Model Size | VRAM Required | Example Models |
|------------|---------------|----------------|
| 3B params  | 6-8GB         | Llama-3.2-3B   |
| 7-8B params| 14-16GB       | Llama-3.1-8B   |
| 13B params | 26-30GB       | Llama-2-13B    |
| 70B params | 140GB+        | Llama-3.1-70B  |

**CPU Usage**: Possible but 10-100x slower than GPU. Use small models only.

## Performance Characteristics

### Model Loading
- **First run**: 5-15 minutes (HuggingFace download)
- **Subsequent runs**: 30-60 seconds (loading from cache)
- **After first agent**: Instant (model cached in memory)

### Inference Speed (per agent decision)
- **vLLM (GPU)**: 0.5-2 seconds
- **Transformers (GPU)**: 2-10 seconds
- **Transformers (CPU)**: 30-300 seconds (not recommended)

### Memory Usage
- Model loaded once, shared by all agents
- Additional VRAM per agent: ~100-500MB (conversation history)

## Known Limitations

1. **vLLM GPU-only**: Falls back to Transformers on CPU
2. **No streaming**: Responses generated in full (not incrementally)
3. **No quantization**: Full precision models (future enhancement)
4. **Single GPU**: No automatic multi-GPU support (manual tensor parallelism via vLLM config)

## Future Enhancements

### High Priority
- [ ] Add quantization support (4-bit, 8-bit) for larger models
- [ ] Implement batch inference for multiple agents
- [ ] Add progress bars for model download
- [ ] Support for LoRA adapters

### Medium Priority
- [ ] Multi-GPU automatic distribution
- [ ] Model warmup optimization
- [ ] Caching compiled models
- [ ] Support for GGUF format models

### Low Priority
- [ ] Streaming responses
- [ ] Custom stopping criteria
- [ ] Fine-tuning integration
- [ ] Model benchmarking utility

## Compatibility

### Python Version
- Tested: Python 3.10
- Expected: Python 3.8+

### Dependencies
All required packages already in `environment.yaml`:
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace Transformers library
- `vllm` - Fast inference engine (optional but recommended)

### Operating Systems
- ✓ Linux (tested, recommended)
- ✓ macOS (should work, may need MPS device support)
- ? Windows (untested, may require adjustments)

## Troubleshooting

See `examples/README.md` for comprehensive troubleshooting guide.

### Quick Fixes
- **Out of memory**: Use smaller model or set `max_parallel=1`
- **Slow inference**: Ensure GPU is being used, check `nvidia-smi`
- **Import errors**: Verify vLLM is installed, or set `use_vllm=False`
- **Download fails**: Check HuggingFace hub status, retry

## Contact

For issues specific to local model implementation:
1. Check `examples/README.md` troubleshooting section
2. Run `test_local_model.py` to identify the issue
3. Open GitHub issue with error details

## Acknowledgments

Implementation follows SwarmBench's existing architecture and maintains compatibility with the original API-based workflow. Local model support uses standard HuggingFace tools (Transformers, vLLM) for maximum compatibility and community support.
