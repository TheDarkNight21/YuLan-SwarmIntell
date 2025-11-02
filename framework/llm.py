from openai import AsyncOpenAI
import asyncio
import torch
from typing import Optional
import warnings
import concurrent.futures


class Chat:
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False):
        self.model = model
        self.msg = []
        self.memory = memory
        self.sys_prompt = {'role': 'system', 'content': ''}
        self.stream = stream
        self.api_key = api_key
        self.base_url = base_url
        self.usage = 0

    def _gen_wrapper(self, response):
        tmp = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                s = chunk.choices[0].delta.content
                tmp.append(s)
                yield s
        self.msg.append({'role': 'assistant', 'content': ''.join(tmp)})

    def _format_response(self, response):
        if self.stream:
            return self._gen_wrapper(response)
        else:
            s = response.choices[0].message.content
            self.msg.append({'role': 'assistant', 'content': s})
            return s

    def jump_back(self):
        while len(self.msg) > 0 and self.msg[-1]['role'] == 'assistant':
            self.msg = self.msg[:-1]
        if len(self.msg) > 0:
            self.msg = self.msg[:-1]

    def system(self, content):
        self.sys_prompt = {'role': 'system', 'content': content}

    async def generate(self, content):
        self.msg.append({'role': 'user', 'content': content})
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[self.sys_prompt] + (self.msg if self.memory else self.msg[-1:]),  # 包含系统提示和消息历史
            stream=self.stream,
            timeout=3600,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        self.usage += response.usage.completion_tokens
        await client.close()

        return self._format_response(response)


class UserAgent:
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False):
        self.system_prompt = ''
        self.usage = 0

    async def generate(self, content):
        print('SYSTEM')
        print(self.system_prompt)
        print('PROMPT')
        print(content)
        return input('>>>')

    def system(self, content):
        self.system_prompt = content


# Global cache for loaded models to avoid reloading
_MODEL_CACHE = {}


class LocalChat:
    """
    Local model inference using either vLLM (faster, GPU) or Transformers (universal).
    Compatible interface with Chat class for drop-in replacement.
    """

    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 512,
                 temperature: float = 0.7, use_vllm: bool = True, stream: bool = False,
                 memory: bool = False):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_vllm = use_vllm
        self.stream = stream
        self.memory = memory
        self.msg = []
        self.sys_prompt = {'role': 'system', 'content': ''}
        self.usage = 0

        # Create thread pool executor for Windows compatibility
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Load model (with caching)
        self.model, self.tokenizer, self.backend = self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load model using vLLM or Transformers, with caching."""
        cache_key = f"{self.model_path}_{self.device}_{self.use_vllm}"

        if cache_key in _MODEL_CACHE:
            print(f"[LocalChat] Using cached model: {self.model_path}")
            return _MODEL_CACHE[cache_key]

        print(f"[LocalChat] Loading model: {self.model_path} on {self.device}...")

        # Disable vLLM on Windows (not supported)
        import platform
        if platform.system() == "Windows" and self.use_vllm:
            warnings.warn("vLLM is not supported on Windows. Using Transformers backend.")
            self.use_vllm = False

        # Try vLLM first if requested and on GPU
        if self.use_vllm and self.device == "cuda":
            try:
                from vllm import LLM, SamplingParams
                print("[LocalChat] Using vLLM backend for faster inference...")
                model = LLM(
                    model=self.model_path,
                    trust_remote_code=True,
                    dtype="auto",
                    tensor_parallel_size=1,
                )
                _MODEL_CACHE[cache_key] = (model, None, "vllm")
                print(f"[LocalChat] Model loaded successfully with vLLM!")
                return model, None, "vllm"
            except Exception as e:
                warnings.warn(f"vLLM loading failed: {e}. Falling back to Transformers...")

        # Fallback to Transformers
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            print("[LocalChat] Using Transformers backend...")

            print(f"[LocalChat] Loading tokenizer from {self.model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            print(f"[LocalChat] Tokenizer loaded successfully")

            print(f"[LocalChat] Loading model weights (this may take 30-60 seconds)...")

            # Load model - handle device_map carefully for Windows/accelerate compatibility
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            # Only use device_map if on CUDA (requires accelerate library)
            if self.device == "cuda":
                try:
                    # Try with device_map first (requires accelerate)
                    load_kwargs["device_map"] = "cuda"
                    model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
                except ImportError as e:
                    if "accelerate" in str(e).lower():
                        warnings.warn(f"accelerate not installed, loading without device_map (slower). Install with: pip install accelerate")
                        # Fallback: load without device_map
                        del load_kwargs["device_map"]
                        model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
                        print(f"[LocalChat] Moving model to CUDA manually...")
                        model = model.to(self.device)
                    else:
                        raise
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

            print(f"[LocalChat] Model weights loaded")

            if self.device == "cpu":
                print(f"[LocalChat] Moving model to CPU...")
                model = model.to(self.device)

            _MODEL_CACHE[cache_key] = (model, tokenizer, "transformers")
            print(f"[LocalChat] ✓ Model loaded successfully with Transformers!")
            return model, tokenizer, "transformers"

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}")

    def _format_messages_for_model(self, messages):
        """Format messages using the model's chat template if available."""
        if self.backend == "transformers" and self.tokenizer is not None:
            # Try to use chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    return formatted
                except Exception as e:
                    warnings.warn(f"Chat template failed: {e}. Using simple format.")

        # Fallback: Simple format
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"System: {content}\n\n"
            elif role == 'user':
                formatted += f"User: {content}\n\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant:"
        return formatted

    async def generate(self, content: str) -> str:
        """Generate response using local model (async wrapper for sync inference)."""
        print(f"[LocalChat] generate() called, preparing inference...")
        self.msg.append({'role': 'user', 'content': content})

        # Prepare messages
        messages = [self.sys_prompt] + (self.msg if self.memory else self.msg[-1:])

        # Run inference in thread pool to avoid blocking
        # Use run_in_executor for Windows compatibility instead of asyncio.to_thread
        print(f"[LocalChat] Running inference in thread pool...")
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            messages
        )
        print(f"[LocalChat] Inference complete")

        self.msg.append({'role': 'assistant', 'content': response_text})
        return response_text

    def _generate_sync(self, messages) -> str:
        """Synchronous generation (called in thread pool)."""
        print(f"[LocalChat._generate_sync] Starting sync generation with {self.backend} backend")
        if self.backend == "vllm":
            return self._generate_vllm(messages)
        else:
            return self._generate_transformers(messages)

    def _generate_vllm(self, messages) -> str:
        """Generate using vLLM."""
        from vllm import SamplingParams

        # Format prompt
        prompt = self._format_messages_for_model(messages)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=0.9,
        )

        # Generate
        outputs = self.model.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        # Track token usage
        self.usage += len(outputs[0].outputs[0].token_ids)

        return response_text

    def _generate_transformers(self, messages) -> str:
        """Generate using Transformers."""
        print(f"[LocalChat._generate_transformers] Formatting prompt...")
        # Format prompt
        prompt = self._format_messages_for_model(messages)

        print(f"[LocalChat._generate_transformers] Tokenizing...")
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        print(f"[LocalChat._generate_transformers] Generating response (this may take 10-30 seconds)...")
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        print(f"[LocalChat._generate_transformers] Decoding response...")
        # Decode (only new tokens)
        response_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Track token usage
        self.usage += outputs.shape[1] - inputs['input_ids'].shape[1]

        print(f"[LocalChat._generate_transformers] ✓ Generation complete")
        return response_text

    def system(self, content: str):
        """Set system prompt."""
        self.sys_prompt = {'role': 'system', 'content': content}

    def jump_back(self):
        """Remove last exchange from history (for retries)."""
        while len(self.msg) > 0 and self.msg[-1]['role'] == 'assistant':
            self.msg = self.msg[:-1]
        if len(self.msg) > 0:
            self.msg = self.msg[:-1]
