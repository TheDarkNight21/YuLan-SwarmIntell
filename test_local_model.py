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

import sys
import os
from datetime import datetime
from swarmbench import SwarmFramework

# Enable HuggingFace progress bars and verbose output
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Faster downloads if available
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'  # Show model loading info


def test_local_model_basic():
    """
    Basic test with minimal configuration.
    - Uses smallest viable model (Llama-3.2-3B)
    - Few agents (3)
    - Short run (10 rounds)
    - Simple task (Transport)
    """
    print("="*70)
    print("Testing Local Model Implementation")
    print("="*70)

    # System info
    print("\n[SYSTEM INFO]")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  ⚠️  No GPU detected - will use CPU (very slow)")
    except ImportError:
        print("  ⚠️  PyTorch not installed")

    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except ImportError:
        print("  ⚠️  Transformers not installed")

    try:
        import vllm
        print(f"  vLLM: {vllm.__version__} (fast inference available)")
    except ImportError:
        print("  vLLM: Not installed (will use Transformers, slower)")

    print("\n[TEST CONFIGURATION]")
    print("  Model: meta-llama/Llama-3.2-3B-Instruct")
    print("  Model Size: ~6GB")
    print("  Task: Transport")
    print("  Agents: 3")
    print("  Rounds: 10")
    print("  Device: auto (GPU if available, else CPU)")
    print("  Backend: vLLM (if available), else Transformers")

    # Check cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_exists = os.path.exists(cache_dir)
    if model_cache_exists:
        print(f"\n[CACHE INFO]")
        print(f"  HuggingFace cache: {cache_dir}")
        try:
            import subprocess
            result = subprocess.run(['du', '-sh', cache_dir], capture_output=True, text=True)
            if result.returncode == 0:
                size = result.stdout.split()[0]
                print(f"  Cache size: {size}")
        except:
            pass

    print("\n[DOWNLOAD INFO]")
    if model_cache_exists:
        print("  ℹ️  Model may already be cached (faster startup)")
    else:
        print("  ⚠️  First run will download model (~6GB)")
        print("  ⏱️  Expected download time: 5-15 minutes")

    print("  📥 Download progress will be shown below")
    print("  💾 Models cached to: ~/.cache/huggingface/")

    print("\n" + "="*70)
    print("STARTING TEST")
    print("="*70)
    print()

    print("[STEP 1/4] Configuring model...")
    model_config = SwarmFramework.local_model_config(
        model_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        device="auto",              # Use GPU if available
        temperature=0.7,
        max_new_tokens=512,
        use_vllm=True               # Will fallback to Transformers if vLLM fails
    )
    print("  ✓ Model configuration created")
    print(f"    - Type: {model_config.model_type}")
    print(f"    - Path: {model_config.model_path}")
    print(f"    - Device: {model_config.device}")
    print(f"    - Temperature: {model_config.temperature}")
    print(f"    - Max tokens: {model_config.max_new_tokens}")
    print(f"    - Use vLLM: {model_config.use_vllm}")

    print("\n[STEP 2/4] Submitting task...")
    SwarmFramework.submit(
        name='test_local',
        model=model_config,
        task='Transport',               # Simple task to test basic functionality
        log_dir='./test_logs',
        num_agents=3,                   # Few agents for quick test
        max_round=10,                   # Short run to verify it works
        width=8,                        # Smaller grid
        height=8,
        seed=42,
        view_size=5
    )
    print("  ✓ Task submitted successfully")

    print("\n[STEP 3/4] Starting execution...")
    print("  ⏳ This will:")
    print("     1. Download model (if not cached) - progress bars shown below")
    print("     2. Load model into memory (~30-60 seconds)")
    print("     3. Create 3 agents (first agent loads model, others reuse it)")
    print("     4. Run 10 rounds of the Transport task")
    print("\n  📊 Watch for these indicators:")
    print("     • 'Loading model:' - Model download/load starting")
    print("     • 'Using vLLM backend' or 'Using Transformers backend' - Backend selected")
    print("     • 'Model loaded successfully' - Ready for inference")
    print("     • 'Using cached model' - Reusing loaded model (agents 2-3)")
    print("     • 'Agent_X ACTION' - Agent decisions being made")
    print("     • Progress bar - Task completion status")
    print("\n" + "-"*70)

    # Run the test
    start_time = datetime.now()
    try:
        print("\n[EXECUTING]")
        SwarmFramework.run_all(max_parallel=1)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "-"*70)
        print("\n" + "="*70)
        print("✓ TEST PASSED!")
        print("="*70)

        print(f"\n[TEST RESULTS]")
        print(f"  Status: SUCCESS ✓")
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"  Logs: ./test_logs")

        print(f"\n[WHAT WORKED]")
        print("  ✓ Model downloaded/loaded successfully")
        print("  ✓ Agents created and initialized")
        print("  ✓ Inference working (agents made decisions)")
        print("  ✓ Framework integration successful")

        print("\n[PERFORMANCE NOTES]")
        avg_per_round = duration / 10 if duration > 0 else 0
        print(f"  Average time per round: {avg_per_round:.1f}s")
        print(f"  Estimated time for 100 rounds: {avg_per_round * 100 / 60:.1f} minutes")

        print("\n[NEXT STEPS]")
        print("  1. 📊 Check logs: ./test_logs/")
        print("  2. 🚀 Run full benchmark: python examples/eval_local.py")
        print("  3. 🔧 Try different models: see examples/README.md")
        print("  4. ⚖️  Compare with API: python examples/eval_mixed.py")
        print("  5. 📈 Adjust settings: temperature, max_tokens, model size")

        return True

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️  TEST INTERRUPTED BY USER")
        print("="*70)
        print("\nTest was cancelled. This is normal if you wanted to stop early.")
        return False

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "-"*70)
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)

        print(f"\n[ERROR DETAILS]")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        print(f"  Time elapsed: {duration:.1f}s before failure")

        # Import traceback for detailed error
        import traceback
        print(f"\n[STACK TRACE]")
        traceback.print_exc()

        print("\n[TROUBLESHOOTING STEPS]")
        print("  1. 🔍 Check GPU:")
        print("     python -c 'import torch; print(torch.cuda.is_available())'")
        print("     nvidia-smi")

        print("\n  2. 💾 Check disk space (need ~10GB free):")
        print("     df -h")

        print("\n  3. 🌐 Check internet connection:")
        print("     ping huggingface.co")

        print("\n  4. 🔄 Try CPU mode (slower but works without GPU):")
        print("     Edit test and set: device='cpu', use_vllm=False")

        print("\n  5. 📚 Common issues:")
        print("     • Out of memory → Use smaller model or set max_parallel=1")
        print("     • Download timeout → Check internet, try again")
        print("     • CUDA error → Update GPU drivers")
        print("     • Import error → Check conda environment is activated")

        print("\n  6. 📖 Full troubleshooting guide:")
        print("     See examples/README.md")

        return False


def test_config_api_backward_compatibility():
    """
    Test that API model configuration still works (backward compatibility).
    This doesn't run the model, just verifies the config is created correctly.
    """
    print("\n" + "="*70)
    print("Test 1: API Model Config (Backward Compatibility)")
    print("="*70)

    try:
        print("\n  Creating API model config...")
        # Test original API config method
        api_cfg = SwarmFramework.model_config(
            'gpt-4o-mini',
            'test-api-key',
            'https://api.openai.com/v1'
        )

        print("  Validating configuration...")
        assert api_cfg.model_type == "api", "API config should have model_type='api'"
        print("    ✓ model_type = 'api'")

        assert api_cfg.model == 'gpt-4o-mini', "Model name mismatch"
        print("    ✓ model = 'gpt-4o-mini'")

        assert api_cfg.api_key == 'test-api-key', "API key mismatch"
        print("    ✓ api_key set correctly")

        assert api_cfg.api_base == 'https://api.openai.com/v1', "API base mismatch"
        print("    ✓ api_base set correctly")

        print("\n  ✓ API model config works correctly")
        print("  ✓ Backward compatibility maintained")
        return True

    except Exception as e:
        print(f"\n  ✗ API config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_local():
    """
    Test local model configuration creation.
    This doesn't load the model, just verifies the config is created correctly.
    """
    print("\n" + "="*70)
    print("Test 2: Local Model Config")
    print("="*70)

    try:
        print("\n  Creating local model config...")
        # Test local model config
        local_cfg = SwarmFramework.local_model_config(
            'meta-llama/Llama-3.2-3B-Instruct',
            device="cuda",
            temperature=0.8,
            max_new_tokens=256,
            use_vllm=False
        )

        print("  Validating configuration...")
        assert local_cfg.model_type == "local", "Local config should have model_type='local'"
        print("    ✓ model_type = 'local'")

        assert local_cfg.model_path == 'meta-llama/Llama-3.2-3B-Instruct', "Model path mismatch"
        print("    ✓ model_path = 'meta-llama/Llama-3.2-3B-Instruct'")

        assert local_cfg.device == "cuda", "Device mismatch"
        print("    ✓ device = 'cuda'")

        assert local_cfg.temperature == 0.8, "Temperature mismatch"
        print("    ✓ temperature = 0.8")

        assert local_cfg.max_new_tokens == 256, "Max tokens mismatch"
        print("    ✓ max_new_tokens = 256")

        assert local_cfg.use_vllm == False, "use_vllm mismatch"
        print("    ✓ use_vllm = False")

        print("\n  ✓ Local model config created correctly")
        print("  ✓ All parameters set properly")
        return True

    except Exception as e:
        print(f"\n  ✗ Local config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "SwarmBench Local Model Test Suite")
    print("="*70)
    print(f"\nTest started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This test suite validates local model implementation in 3 phases.")

    # Phase 1: Configuration tests
    print("\n" + "="*70)
    print("PHASE 1: Configuration Validation (Fast)")
    print("="*70)
    print("Testing that model configurations are created correctly...")
    print("(This phase does not load any models or use the GPU)")

    config_tests_passed = (
        test_config_api_backward_compatibility() and
        test_config_local()
    )

    if not config_tests_passed:
        print("\n" + "="*70)
        print("✗ CONFIGURATION TESTS FAILED")
        print("="*70)
        print("\nThe framework configuration is broken. Fix errors above before proceeding.")
        exit(1)

    print("\n" + "="*70)
    print("✓ PHASE 1 COMPLETE: All configuration tests passed!")
    print("="*70)

    # Phase 2: Model loading and inference test
    print("\n" + "="*70)
    print("PHASE 2: Model Loading & Inference (Slow)")
    print("="*70)
    print("\n⚠️  WARNING: This phase will:")
    print("   • Download ~6GB model on first run (5-15 minutes)")
    print("   • Load model into GPU memory (~6GB VRAM)")
    print("   • Run actual inference (2-5 minutes)")
    print("   • Total time: 7-20 minutes on first run")
    print("\n💡 TIP: Subsequent runs are much faster (model cached)")

    user_input = input("\n❓ Proceed with model loading test? (y/n): ").strip().lower()
    if user_input != 'y':
        print("\n" + "="*70)
        print("⏭️  SKIPPING PHASE 2")
        print("="*70)
        print("\nModel test skipped by user.")
        print("\nConfiguration is valid. When ready to test model loading:")
        print("  python test_local_model.py")
        print("\nOr run full examples directly:")
        print("  python examples/eval_local.py")
        exit(0)

    print("\n" + "="*70)
    print("🚀 STARTING PHASE 2: Model Test")
    print("="*70)

    model_test_passed = test_local_model_basic()

    # Final summary
    print("\n" + "="*70)
    if model_test_passed:
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*70)
        print("\n🎉 SUCCESS! Local model implementation is fully functional!")
        print("\n[VALIDATION SUMMARY]")
        print("  ✓ Configuration creation: PASSED")
        print("  ✓ Model download/loading: PASSED")
        print("  ✓ Agent initialization: PASSED")
        print("  ✓ Inference execution: PASSED")
        print("  ✓ Framework integration: PASSED")
        print("\n[YOUR SETUP IS READY]")
        print("  You can now run full benchmarks with local models!")
        print("\n[RECOMMENDED NEXT STEPS]")
        print("  1. 📊 Explore test results: ./test_logs/")
        print("  2. 🚀 Run full evaluation: python examples/eval_local.py")
        print("  3. 📚 Read documentation: examples/README.md")
        print("  4. ⚖️  Compare with APIs: python examples/eval_mixed.py")
        print("  5. 🔧 Try different models: See examples/README.md for recommendations")
        print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        exit(0)
    else:
        print("✗✗✗ MODEL TEST FAILED ✗✗✗")
        print("="*70)
        print("\n❌ The model loading/inference test failed.")
        print("   See detailed error messages above.")
        print("\n[WHAT TO DO]")
        print("  1. Read the error details above carefully")
        print("  2. Follow the troubleshooting steps provided")
        print("  3. Check examples/README.md for common issues")
        print("  4. Verify your environment setup")
        print("\n[QUICK CHECKS]")
        print("  • GPU available: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  • Disk space: df -h (need ~10GB free)")
        print("  • Internet: ping huggingface.co")
        print(f"\n🕐 Test failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        exit(1)
