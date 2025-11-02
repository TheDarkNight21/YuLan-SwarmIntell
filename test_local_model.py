"""
Simple test script to verify local model implementation.

This script tests the basic functionality of local model support with a minimal setup.
Run this in your GPU environment to verify everything works before running full benchmarks.

Usage:
    python test_local_model.py

Requirements:
- GPU recommended (will work on CPU but very slow)
- At least 6GB VRAM for the test model
- Internet connection for first-time model download
"""

from swarmbench import SwarmFramework


def test_local_model_basic():
    """
    Basic test with minimal configuration.
    - Uses smallest viable model (Llama-3.2-3B)
    - Few agents (3)
    - Short run (10 rounds)
    - Simple task (Transport)
    """
    print("="*60)
    print("Testing Local Model Implementation")
    print("="*60)
    print("\nConfiguration:")
    print("  Model: meta-llama/Llama-3.2-3B-Instruct")
    print("  Task: Transport")
    print("  Agents: 3")
    print("  Rounds: 10")
    print("  Device: auto (GPU if available)")
    print("\nNote: First run will download the model (~6GB)")
    print("      This may take 5-15 minutes depending on your connection")
    print("="*60)
    print()

    # Submit test task with local model
    SwarmFramework.submit(
        name='test_local',
        model=SwarmFramework.local_model_config(
            model_path='meta-llama/Llama-3.2-3B-Instruct',
            device="auto",              # Use GPU if available
            temperature=0.7,
            max_new_tokens=512,
            use_vllm=True               # Will fallback to Transformers if vLLM fails
        ),
        task='Transport',               # Simple task to test basic functionality
        log_dir='./test_logs',
        num_agents=3,                   # Few agents for quick test
        max_round=10,                   # Short run to verify it works
        width=8,                        # Smaller grid
        height=8,
        seed=42,
        view_size=5
    )

    print("Starting test run...")
    print("Watch for:")
    print("  ✓ Model loading messages")
    print("  ✓ Agent decisions being made")
    print("  ✓ Actions printed (e.g., 'Agent_0 UP')")
    print("  ✓ Progress updates")
    print()

    # Run the test
    try:
        SwarmFramework.run_all(max_parallel=1)
        print("\n" + "="*60)
        print("✓ TEST PASSED!")
        print("="*60)
        print("\nLocal model implementation is working correctly!")
        print("Check ./test_logs for detailed results")
        print("\nNext steps:")
        print("  1. Run examples/eval_local.py for full benchmark")
        print("  2. Try different models (see examples/README.md)")
        print("  3. Compare with API models using examples/eval_mixed.py")
        return True

    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Check GPU availability: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  2. Verify VRAM: nvidia-smi")
        print("  3. Check disk space: df -h")
        print("  4. Try CPU mode (slow): device='cpu', use_vllm=False")
        print("\nSee examples/README.md for more troubleshooting tips")
        return False


def test_config_api_backward_compatibility():
    """
    Test that API model configuration still works (backward compatibility).
    This doesn't run the model, just verifies the config is created correctly.
    """
    print("\n" + "="*60)
    print("Testing API Model Config (Backward Compatibility)")
    print("="*60)

    try:
        # Test original API config method
        api_cfg = SwarmFramework.model_config(
            'gpt-4o-mini',
            'test-api-key',
            'https://api.openai.com/v1'
        )

        assert api_cfg.model_type == "api", "API config should have model_type='api'"
        assert api_cfg.model == 'gpt-4o-mini', "Model name mismatch"
        assert api_cfg.api_key == 'test-api-key', "API key mismatch"
        assert api_cfg.api_base == 'https://api.openai.com/v1', "API base mismatch"

        print("✓ API model config works correctly")
        print("✓ Backward compatibility maintained")
        return True

    except Exception as e:
        print(f"✗ API config test failed: {e}")
        return False


def test_config_local():
    """
    Test local model configuration creation.
    This doesn't load the model, just verifies the config is created correctly.
    """
    print("\n" + "="*60)
    print("Testing Local Model Config")
    print("="*60)

    try:
        # Test local model config
        local_cfg = SwarmFramework.local_model_config(
            'meta-llama/Llama-3.2-3B-Instruct',
            device="cuda",
            temperature=0.8,
            max_new_tokens=256,
            use_vllm=False
        )

        assert local_cfg.model_type == "local", "Local config should have model_type='local'"
        assert local_cfg.model_path == 'meta-llama/Llama-3.2-3B-Instruct', "Model path mismatch"
        assert local_cfg.device == "cuda", "Device mismatch"
        assert local_cfg.temperature == 0.8, "Temperature mismatch"
        assert local_cfg.max_new_tokens == 256, "Max tokens mismatch"
        assert local_cfg.use_vllm == False, "use_vllm mismatch"

        print("✓ Local model config created correctly")
        print("✓ All parameters set properly")
        return True

    except Exception as e:
        print(f"✗ Local config test failed: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SwarmBench Local Model Test Suite")
    print("="*60)

    # Run configuration tests first (fast, no model loading)
    print("\n[1/3] Testing configuration creation...")
    config_tests_passed = (
        test_config_api_backward_compatibility() and
        test_config_local()
    )

    if not config_tests_passed:
        print("\n✗ Configuration tests failed. Fix these before proceeding.")
        exit(1)

    print("\n" + "="*60)
    print("Configuration tests passed!")
    print("="*60)

    # Run actual model test (slow, downloads and loads model)
    print("\n[2/3] Testing actual model loading and inference...")
    print("This will take several minutes on first run (model download)")

    user_input = input("\nProceed with model test? (y/n): ").strip().lower()
    if user_input != 'y':
        print("\nSkipping model test.")
        print("Run this script again when ready to test model loading.")
        exit(0)

    model_test_passed = test_local_model_basic()

    if model_test_passed:
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nYour local model implementation is working correctly.")
        print("You can now run the full benchmarks in examples/")
        exit(0)
    else:
        print("\n" + "="*60)
        print("MODEL TEST FAILED ✗")
        print("="*60)
        print("\nSee error messages above for troubleshooting.")
        exit(1)
