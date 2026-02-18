import os
import sys
import argparse
import torch
from stable_baselines3 import PPO


from PentestGymEnv import PentestGymEnv
from colorama import Fore, init

init(autoreset=True)

def run_inference(model_path, target_url, max_steps=10):
    """
    Trained model'i yükleyip belirli bir hedef üzerinde koşturur.
    """
    if not os.path.exists(model_path):
        print(f"{Fore.RED}[ERROR] Model dosyası bulunamadı: {model_path}")
        return

    print(f"{Fore.CYAN}[*] Ana Model yükleniyor: {model_path}")
    
    # Env'i tek bir hedefle başlatıyoruz
    env = PentestGymEnv(targets=[target_url])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = PPO.load(model_path, env=env, device=device)
        print(f"{Fore.GREEN}[+] Ana Model başarıyla yüklendi.")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Ana Model yükleme hatası: {e}")
        return



    obs, info = env.reset()
    total_reward = 0
    
    for step in range(1, max_steps + 1):
        # Karar verme aşamasında adapter varsa ve aksiyon 4 ise adapter'ı kullanabiliriz
        # Ya da her zaman model'den tahmin alıp, aksiyon 4 ise adapter'dan tekrar tahmin isteyebiliriz
        action, _states = model.predict(obs, deterministic=True)
        
        # Action isimlerini mapleyelim
        action_names = {
            0: "Crawling",
            1: "Tech Detection",
            2: "CVE Lookup",
            3: "Exploit Verification",
            4: "Source Analysis",
            5: "Advanced Discovery",
            6: "Deep Scan"
        }
        
        curr_action = action_names.get(int(action), f"Unknown({action})")
        
        # MODÜLER YOL: Eğer aksiyon Source Analysis ise ve adapter varsa, 
        # adapter'ın bu durumdaki 'uzman' görüşünü alalım

        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"{Fore.GREEN}[!] İşlem tamamlandı veya durduruldu.")
            break

    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.GREEN}[DONE] Tarama Bitti. Toplam Ödül: {total_reward}")
    
    final_findings = env.agent.findings
    print(f"{Fore.GREEN}[✓] Rapor oluşturuldu. Ajan durumu: {final_findings.get('scan_status')}")
    print(f"{Fore.CYAN}{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trained Pentest AI Agent Inference")
    parser.add_argument("--model", type=str, default="logs/checkpoints/rl_model_700_steps.zip", help="Path to the .zip model checkpoint")
    parser.add_argument("--target", type=str, default="http://localhost:8080", help="Target URL to pentest")
    parser.add_argument("--steps", type=int, default=10, help="Maximum steps for the agent")

    args = parser.parse_args()
    
    run_inference(args.model, args.target, args.steps)
