import os
import sys
import argparse
import torch
from stable_baselines3 import PPO
from PentestGymEnv import PentestGymEnv
from colorama import Fore, init

init(autoreset=True)

def run_inference(model_path, target_url, max_steps=10, adapter_path=None):
    """
    Trained model'i yükleyip belirli bir hedef üzerinde koşturur.
    Opsiyonel olarak bir 'Code Expert' adapter'ı yükler.
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

    # Adapter Yükleme
    adapter = None
    if adapter_path and os.path.exists(adapter_path):
        try:
            print(f"{Fore.MAGENTA}[*] Code Expert Adapter yükleniyor: {adapter_path}")
            adapter = PPO.load(adapter_path, env=env, device=device)
            print(f"{Fore.GREEN}[+] Adapter başarıyla yüklendi.")
        except Exception as e:
            print(f"{Fore.YELLOW}[!] Adapter yüklenemedi: {e}")

    print(f"{Fore.YELLOW}[*] Ajan başlatılıyor. Hedef: {target_url}")
    print(f"{Fore.YELLOW}[*] Maksimum adım sayısı: {max_steps}")

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
        if int(action) == 4 and adapter:
            print(f"{Fore.MAGENTA}[ADAPTER] Code Expert devreye giriyor...")
            # Burada adapter'ın kendi prediction'ını alabiliriz veya direkt devam edebiliriz.
            # Şimdilik adapter yüklü ise aksiyonu adapter'dan geçirelim.
            action, _ = adapter.predict(obs, deterministic=True)
            curr_action = action_names.get(int(action), f"Unknown({action})")

        print(f"{Fore.BLUE}[Step {step}] Ajan aksiyon alıyor: {Fore.WHITE}{curr_action}")
        
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
    parser.add_argument("--model", type=str, required=True, help="Path to the .zip model checkpoint")
    parser.add_argument("--target", type=str, required=True, help="Target URL to pentest")
    parser.add_argument("--steps", type=int, default=10, help="Maximum steps for the agent")
    parser.add_argument("--adapter", type=str, default=None, help="Path to the Code Expert adapter (.zip)")

    args = parser.parse_args()
    
    run_inference(args.model, args.target, args.steps, args.adapter)
