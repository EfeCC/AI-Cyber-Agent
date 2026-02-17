import sys
import os
import torch
import yaml
from stable_baselines3 import PPO
from PentestGymEnv import PentestGymEnv
from colorama import Fore, init

init(autoreset=True)

# Add PentestAiAgent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PentestAiAgent')))

class CodeExpertTrainer:
    def __init__(self, base_model_path="logs/checkpoints/rl_model_800_steps.zip", config_path='docker-config.yaml'):
        self.base_model_path = base_model_path
        self.config_path = config_path
        self.training_targets = []
        self.load_docker_config()

    def load_docker_config(self):
        """Load targets from docker-config.yaml"""
        if not os.path.exists(self.config_path):
            print(f"{Fore.RED}[ERROR] {self.config_path} not found.")
            return

        with open(self.config_path, 'r') as f:
            docker_conf = yaml.safe_load(f)
        
        services = docker_conf.get('services', {})
        print(f"{Fore.CYAN}[*] Loading targets for Code Expert training...")
        
        for service_name, service_conf in services.items():
            ports = service_conf.get('ports', [])
            for port_mapping in ports:
                host_port = port_mapping.split(':')[0]
                target_url = f"http://localhost:{host_port}"
                self.training_targets.append(target_url)
                print(f"  Added Target: {target_url} ({service_name})")

    def train_adapter(self, total_timesteps=1024):
        if not os.path.exists(self.base_model_path):
            print(f"{Fore.RED}[ERROR] Base model not found at {self.base_model_path}")
            return

        print(f"\n{Fore.YELLOW}[1/3] Initializing Env & Loading Base Model...")
        env = PentestGymEnv(targets=self.training_targets)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(self.base_model_path, env=env, device=device)
        
        print(f"{Fore.YELLOW}[2/3] Freezing Base Model (LoRA-style)...")
        # In a standard MlpPolicy, the policy and value networks are what we want to adapt.
        # However, for a true LoRA approach, we would freeze everything and add layers.
        # Here, we will freeze the feature extractor (shared representation) and only train the policy heads.
        
        # Freeze Feature Extractor
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
            
        print(f"{Fore.GREEN}[+] Feature extractor frozen. Only policy/value heads will be trained.")

        print(f"\n{Fore.YELLOW}[3/3] Starting 'Code Expert' training for {total_timesteps} steps...")
        # Reduce learning rate for fine-tuning the adapter
        model.learning_rate = 0.0001 
        
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        
        output_name = "ppo_pentest_agent_code_expert"
        model.save(output_name)
        print(f"\n{Fore.GREEN}[DONE] Code Expert Adapter saved as {output_name}.zip")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Code Expert LoRA Trainer")
    parser.add_argument("--base", type=str, default="logs/checkpoints/rl_model_800_steps.zip", help="Path to base model")
    parser.add_argument("--steps", type=int, default=1024, help="Steps to train the adapter")
    args = parser.parse_args()
    
    trainer = CodeExpertTrainer(base_model_path=args.base)
    trainer.train_adapter(total_timesteps=args.steps)
