"""
Example script for running SwarmBench with local HuggingFace models.

This demonstrates how to use local models instead of API-based models for the benchmark.
Local models are downloaded from HuggingFace and run on your GPU/CPU.

Requirements:
- GPU recommended (CPU is very slow)
- Sufficient VRAM/RAM for the model (3B models need ~6GB VRAM, 7B models need ~14GB)
- Models will be cached in ~/.cache/huggingface/
"""

from swarmbench import SwarmFramework


if __name__ == '__main__':
    # Example 1: Small model (good for testing, fast)
    # Llama-3.2-3B-Instruct is a compact model that works well for most tasks
    SwarmFramework.submit(
        'local_llama_3b',
        SwarmFramework.local_model_config(
            'meta-llama/Llama-3.2-3B-Instruct',  # HuggingFace model ID
            device="auto",          # Use GPU if available, else CPU
            temperature=0.7,        # Sampling temperature (0.0 = greedy, 1.0 = random)
            max_new_tokens=512,     # Max tokens per response
            use_vllm=True           # Use vLLM for faster inference (GPU only)
        ),
        'Transport',                # Task name
        log_dir='./logs',
        num_agents=10,
        max_round=50,               # Reduced for faster testing
        width=10,
        height=10,
        seed=42,
        view_size=5
    )

    # Example 2: Larger model (better performance, slower)
    # Uncomment to test with a 7B parameter model (requires more VRAM)
    # SwarmFramework.submit(
    #     'local_llama_7b',
    #     SwarmFramework.local_model_config(
    #         'meta-llama/Llama-3.1-8B-Instruct',
    #         device="auto",
    #         temperature=0.7,
    #         use_vllm=True
    #     ),
    #     'Pursuit',
    #     log_dir='./logs',
    #     num_agents=10,
    #     max_round=50,
    #     seed=42
    # )

    # Example 3: Using CPU (very slow, not recommended for large runs)
    # Useful for testing on machines without GPU
    # SwarmFramework.submit(
    #     'local_cpu',
    #     SwarmFramework.local_model_config(
    #         'microsoft/Phi-3-mini-4k-instruct',  # Smaller model better for CPU
    #         device="cpu",
    #         temperature=0.7,
    #         use_vllm=False,  # vLLM requires GPU
    #         max_new_tokens=256
    #     ),
    #     'Transport',
    #     log_dir='./logs',
    #     num_agents=5,       # Fewer agents for CPU
    #     max_round=10,       # Much shorter run for CPU
    #     seed=42
    # )

    # Run all submitted tasks
    print("Starting local model evaluation...")
    print("Note: First run will download models from HuggingFace (may take several minutes)")
    print("Models are cached at ~/.cache/huggingface/ for future runs")
    SwarmFramework.run_all(max_parallel=1)  # Run 1 at a time to avoid OOM
    print("Evaluation complete! Check ./logs for results")
