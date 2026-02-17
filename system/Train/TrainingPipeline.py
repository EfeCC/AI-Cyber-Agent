import sys
import os
import time
import json
import yaml
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from PentestGymEnv import PentestGymEnv
import torch

# Add PentestAiAgent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PentestAiAgent')))

try:
    from main import PentestAgent
except ImportError:
    print("Error: Could not import PentestAgent. Make sure PentestAiAgent/main.py exists.")
    sys.exit(1)

class TrainingPipeline:
    def __init__(self, config_path='docker-config.yaml'):
        self.config_path = config_path
        self.training_targets = []
        self.load_docker_config()
        
    def load_docker_config(self):
        """Load targets from docker-config.yaml"""
        if not os.path.exists(self.config_path):
            print(f"Error: {self.config_path} not found.")
            return

        with open(self.config_path, 'r') as f:
            docker_conf = yaml.safe_load(f)
        
        services = docker_conf.get('services', {})
        print(f"Loading targets from {self.config_path}...")
        
        for service_name, service_conf in services.items():
            ports = service_conf.get('ports', [])
            for port_mapping in ports:
                # Assuming format "HOST:CONTAINER"
                host_port = port_mapping.split(':')[0]
                target_url = f"http://localhost:{host_port}"
                self.training_targets.append(target_url)
                print(f"  Added Target: {target_url} ({service_name})")

    def train(self, total_timesteps=1000, load_path=None):
        if not self.training_targets:
            print("No targets found. Exiting.")
            return

        print(f"\n[INIT] Initializing PentestGymEnv with {len(self.training_targets)} targets...")
        env = PentestGymEnv(targets=self.training_targets)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] Training on {device.upper()}")
        
        if load_path and os.path.exists(load_path):
            print(f"[LOAD] Loading existing model from {load_path}...")
            model = PPO.load(load_path, env=env, device=device)
        else:
            print(f"[MODEL] Initializing new PPO Agent...")
            model = PPO("MlpPolicy", env, verbose=1, device=device)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100,
            save_path="./logs/checkpoints/",
            name_prefix="rl_model"
        )
        
        print(f"[TRAIN] Starting training for {total_timesteps} timesteps (Checkpoints enabled)...")
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
        
        print(f"[SAVE] Saving model to ppo_pentest_agent.zip")
        model.save("ppo_pentest_agent")
        
        print(f"[DONE] Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pentest AI Training Pipeline")
    parser.add_argument("--load", type=str, help="Path to a model checkpoint to resume training from")
    parser.add_argument("--steps", type=int, default=2048, help="Total timesteps for training")
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    
    # Checkpoint'ten devam etmek istersen buraya dosya yolunu yazabilirsin
    # Örn: load_path = "logs/checkpoints/rl_model_100_steps.zip"
    load_path = args.load 
    
    pipeline.train(total_timesteps=args.steps, load_path=args.load)
