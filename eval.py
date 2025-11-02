from swarmbench import SwarmFramework

#
# if __name__ == '__main__':
#     name = 1
#     for task in ('Transport', 'Pursuit'): ########## Task Names, avaliable: {'Transport', 'Pursuit', 'Synchronization', 'Foraging', 'Flocking'}
#         for model in ('gpt-4o-mini','meta-llama/llama-3.1-70b-instruct'): ########### Models
#             for seed in (27, 42):
#                 SwarmFramework.submit(
#                     f'exp_{name}',
#                     SwarmFramework.model_config(model, 'YOUR_API_KEY', 'https://openrouter.ai/api/v1'), ########## API
#                     task,
#                     log_dir='./logs', ########## Logging
#                     num_agents=10,
#                     max_round=100,
#                     width=10,
#                     height=10,
#                     seed=seed,
#                     view_size=5
#                 )
#                 name += 1
#
#     SwarmFramework.run_all(max_parallel=4)



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
