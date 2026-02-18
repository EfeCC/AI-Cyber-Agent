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
    performed_actions = set()
    
    for step in range(1, max_steps + 1):
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        # GUIDED SEQUENCE: En mantıklı profesyonel akışı uygula (1-2-6-3-5-4-7)
        # 0:Crawl, 1:Tech, 5:Discovery, 2:CVE, 4:Source, 3:Exploit, 6:Deep
        for target_action in [0, 1, 5, 2, 4, 3, 6]:
            if target_action not in performed_actions:
                action = target_action
                break

        # LOOP PREVENTION: Eğer model (veya üstteki mantık dışındaki adımlar) 
        # sürekli aynı şeyi yapmaya çalışıyorsa (reward hacking) onu zorla
        if action in performed_actions and len(performed_actions) < 7:
            for alt_action in [0, 1, 2, 3, 5, 6, 4]: 
                if alt_action not in performed_actions:
                    action = alt_action
                    break
        
        performed_actions.add(action)
        
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
    
    # Raporu kaydet
    env.agent._save_report()
    
    final_findings = env.agent.findings
    print(f"{Fore.GREEN}[✓] Rapor oluşturuldu. Ajan durumu: {final_findings.get('scan_status')}")
    print(f"{Fore.CYAN}{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trained Pentest AI Agent Inference")
    parser.add_argument("--model", type=str, default="C:/AI-Cyber-Agent/logs/checkpoints/rl_model_700_steps.zip", help="Path to the .zip model checkpoint")
    parser.add_argument("--target", type=str, default="http://localhost:8080", help="Target URL to pentest")
    parser.add_argument("--steps", type=int, default=10, help="Maximum steps for the agent")

    args = parser.parse_args()
    
    run_inference(args.model, args.target, args.steps)
