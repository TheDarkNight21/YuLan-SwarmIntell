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

            # Create brain based on model type
            if model[i].model_type == "api":
                brain = Chat(model[i].api_base, model[i].api_key, model[i].model)
            elif model[i].model_type == "local":
                brain = LocalChat(
                    model[i].model_path,
                    device=model[i].device,
                    max_new_tokens=model[i].max_new_tokens,
                    temperature=model[i].temperature,
                    use_vllm=model[i].use_vllm
                )
            else:
                raise ValueError(f"Unknown model_type: {model[i].model_type}. Must be 'api' or 'local'.")

            self.agents[agent_name] = agent_cls(agent_name, brain, **agent_args)
        
        self.logger = logger_cls(env_name, **logger_args)

        self.env = env_cls(env_name, self.agents, self.logger, **env_args)

        self.env.start()

        return self.env
