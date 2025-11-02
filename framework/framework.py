from framework.llm import Chat, UserAgent, LocalChat


class ModelConfig:
    model: str
    model_type: str = "api"  # "api" or "local"

    # API fields
    api_base: str = None
    api_key: str = None

    # Local model fields
    model_path: str = None
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    use_vllm: bool = True

    def as_dict(self):
        base_dict = {
            'model': self.model,
            'model_type': self.model_type,
        }

        if self.model_type == "api":
            base_dict.update({
                'api_base': self.api_base,
                'api_key': self.api_key,
            })
        elif self.model_type == "local":
            base_dict.update({
                'model_path': self.model_path,
                'device': self.device,
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'use_vllm': self.use_vllm,
            })

        return base_dict


class Framework:

    def __init__(self):
        self.env = None
        self.agents = None
        self.logger = None

    def start(self, agent_cls, env_cls, logger_cls, env_name, num_agents, model,
              agent_args=None, logger_args=None, env_args=None):
        if agent_args is None:
            agent_args = {}

        if logger_args is None:
            logger_args = {}

        if env_args is None:
            env_args = {}

        if isinstance(model, ModelConfig):
            model = [model for _ in range(num_agents)]
        self.agents = {}
        for i in range(num_agents):
            agent_name = f'Agent_{i}'
            print(f"[Framework.start] Creating agent {i + 1}/{num_agents}: {agent_name}")

            # Create brain based on model type
            if model[i].model_type == "api":
                print(f"[Framework.start] Creating API brain for {agent_name}")
                brain = Chat(model[i].api_base, model[i].api_key, model[i].model)
            elif model[i].model_type == "local":
                print(f"[Framework.start] Creating LocalChat brain for {agent_name}")
                print(f"[Framework.start] This will load the model (may take 30-60s)...")
                brain = LocalChat(
                    model[i].model_path,
                    device=model[i].device,
                    max_new_tokens=model[i].max_new_tokens,
                    temperature=model[i].temperature,
                    use_vllm=model[i].use_vllm
                )
                print(f"[Framework.start] LocalChat brain created successfully for {agent_name}")
            else:
                raise ValueError(f"Unknown model_type: {model[i].model_type}. Must be 'api' or 'local'.")

            print(f"[Framework.start] Creating agent instance for {agent_name}")
            self.agents[agent_name] = agent_cls(agent_name, brain, **agent_args)
            print(f"[Framework.start] ✓ Agent {agent_name} created successfully")

        self.logger = logger_cls(env_name, **logger_args)

        self.env = env_cls(env_name, self.agents, self.logger, **env_args)

        self.env.start()

        return self.env
