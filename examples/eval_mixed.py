"""
Example script showing how to mix API-based and local models in SwarmBench.

This demonstrates the flexibility of the framework - you can compare different
model types in a single run, or use different models for different tasks.
"""

from swarmbench import SwarmFramework


if __name__ == '__main__':
    # API-based model (GPT-4o-mini via OpenRouter)
    SwarmFramework.submit(
        'api_gpt4o',
        SwarmFramework.model_config(
            'gpt-4o-mini',
            'YOUR_API_KEY',  # Replace with your actual API key
            'https://openrouter.ai/api/v1'
        ),
        'Transport',
        log_dir='./logs',
        num_agents=10,
        max_round=50,
        seed=42
    )

    # Local model (Llama-3.2-3B)
    SwarmFramework.submit(
        'local_llama3b',
        SwarmFramework.local_model_config(
            'meta-llama/Llama-3.2-3B-Instruct',
            device="auto",
            temperature=0.7,
            use_vllm=True
        ),
        'Transport',  # Same task for comparison
        log_dir='./logs',
        num_agents=10,
        max_round=50,
        seed=42  # Same seed for fair comparison
    )

    # Another API model (different provider/model)
    SwarmFramework.submit(
        'api_llama_70b',
        SwarmFramework.model_config(
            'meta-llama/llama-3.1-70b-instruct',
            'YOUR_API_KEY',
            'https://openrouter.ai/api/v1'
        ),
        'Pursuit',
        log_dir='./logs',
        num_agents=10,
        max_round=50,
        seed=27
    )

    # Run all tasks (API and local models) in parallel
    # Note: Be careful with max_parallel if using local models (GPU memory limits)
    print("Running mixed API and local model evaluation...")
    print("This will compare performance across different model types!")
    SwarmFramework.run_all(max_parallel=2)
    print("\nEvaluation complete!")
    print("Compare results in ./logs to see performance differences between:")
    print("  - API models (fast, costs money, potentially better quality)")
    print("  - Local models (one-time download, free inference, full control)")
