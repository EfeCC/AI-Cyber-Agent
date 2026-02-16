import sys
import os
import time
import json
import yaml
import requests
from modules.reinforcement_learner import ReinforcementLearner

# Add PentestAiAgent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PentestAiAgent')))

try:
    from main import PentestAgent
except ImportError:
    print("Error: Could not import PentestAgent. Make sure PentestAiAgent/main.py exists.")
    sys.exit(1)

class TrainingPipeline:
    def __init__(self, config_path='docker-config.yaml'):
        self.learner = ReinforcementLearner()
        self.config_path = config_path
        self.training_targets = []
        self.ground_truth = {}
        
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
                
                # Define ground truth based on service name
                self.ground_truth[target_url] = self.get_ground_truth_for_service(service_name)
                print(f"  Added Target: {target_url} ({service_name})")

    def get_ground_truth_for_service(self, service_name):
        """Map service names to expected vulnerabilities"""
        service_name = service_name.lower()
        truth = {}
        
        if 'dvwa' in service_name:
            truth = {
                'sql_injection': True,
                'xss': True,
                'csrf': True,
                'vulnerable_versions': ['PHP', 'MySQL']
            }
        elif 'juice' in service_name or 'shop' in service_name: # juice-shop
             truth = {
                'sql_injection': True,
                'xss': True,
                'jwt_manipulation': True
            }
        elif 'wordpress' in service_name:
             truth = {
                'vulnerable_versions': ['WordPress 5.0.0']
             }
        elif 'tomcat' in service_name:
             truth = {
                 'vulnerable_versions': ['Apache Tomcat 8.5.0']
             }
        elif 'webgoat' in service_name:
             truth = {
                 'sql_injection': True,
                 'xss': True
             }
        
        return truth

    def run_training_epoch(self, epochs=5):
        """Eğitim döngüsü"""
        print(f"[TRAINING] {epochs} epoch başlıyor...")
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            epoch_rewards = 0
            
            for target in self.training_targets:
                print(f"\n[TARGET] {target}")
                
                # Check if target is up
                try:
                    requests.get(target, timeout=2)
                except requests.exceptions.ConnectionError:
                    print(f"  [!] Target {target} is unreachable. Skipping.")
                    continue

                # Agent'ı çalıştır
                try:
                    # Initialize agent
                    agent = PentestAgent(target)
                    
                    # Manually trigger phases to simulate run_scan if it doesn't exist or is split
                    # Based on main.py structure, we might need to call phases individually
                    # But checking main.py, it seems it runs sequentially in __init__ or we need to check how it runs.
                    # Looking at main.py content from previous steps, it has phases as methods.
                    # We will assume we need to call them or there is a run method.
                    # If PentestAgent doesn't have run_scan, we'll implement a helper here.
                    
                    self.run_agent_scan(agent)
                    
                    # Sonuçları değerlendir
                    results = agent.findings
                    reward = self.evaluate_results(target, results)
                    epoch_rewards += reward
                    
                    print(f"[REWARD] {reward} puan")
                    
                except Exception as e:
                    print(f"  [ERROR] scanning {target}: {e}")
                
                # Bekle (rate limiting)
                time.sleep(2)
            
            print(f"\n[EPOCH {epoch + 1} TOPLAM REWARD] {epoch_rewards}")
            
            # İstatistikleri kaydet
            self.save_epoch_stats(epoch, epoch_rewards)
        
        # Final rapor
        self.generate_training_report()
    
    def run_agent_scan(self, agent):
        """Execute agent phases"""
        # Based on main.py structure inferred from PentestGymEnv
        agent._phase_1_crawling()
        agent._phase_2_technology_detection()
        agent._phase_3_cve_lookup()
        agent._phase_4_exploit_verification()
        # agent._phase_5_source_analysis() # Skipping some for speed in training if needed
        # agent._phase_6_advanced_discovery()
        # agent._phase_7_deep_scan()

    def evaluate_results(self, target, results):
        """Sonuçları ground truth ile karşılaştır"""
        reward = 0
        truth = self.ground_truth.get(target, {})
        
        # SQL Injection doğrulaması
        verified_sqli = any(
            'sql' in e.get('type', '').lower() 
            for e in results.get('verified_exploits', [])
        )
        
        if verified_sqli and truth.get('sql_injection'):
            reward += 20  # Doğru tespit
            print(f"  ✓ SQL Injection doğru tespit edildi (+20)")
            
            # Başarılı exploit'i öğren
            for exploit in results.get('verified_exploits', []):
                if 'sql' in exploit.get('type', '').lower():
                    self.learner.learn_from_exploit({
                        'payload': exploit.get('payload'),
                        'technology': 'SQL',
                        'target_pattern': target,
                        'bypass_technique': exploit.get('ai_technique', 'standard')
                    })
        
        elif verified_sqli and not truth.get('sql_injection'):
            reward -= 10  # False positive
            print(f"  ✗ False positive: SQL Injection (-10)")
            self.learner.stats['false_positives'] += 1
        
        # XSS doğrulaması
        verified_xss = any(
            'xss' in e.get('type', '').lower() 
            for e in results.get('verified_exploits', [])
        )
        
        if verified_xss and truth.get('xss'):
            reward += 15
            print(f"  ✓ XSS doğru tespit edildi (+15)")
        
        # CVE tespiti
        critical_cves = [
            c for c in results.get('cves', []) 
            if c.get('severity') == 'CRITICAL'
        ]
        
        if critical_cves:
            reward += len(critical_cves) * 5
            print(f"  ✓ {len(critical_cves)} Critical CVE bulundu (+{len(critical_cves) * 5})")
            
        # Add reward for any verified exploit
        if len(results.get('verified_exploits', [])) > 0:
             self.learner.stats['successful_exploits'] += len(results.get('verified_exploits', []))
        
        return reward
    
    def save_epoch_stats(self, epoch, reward):
        """Epoch istatistiklerini kaydet"""
        stats_file = 'training_stats.json'
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            except:
                stats = {'epochs': []}
        else:
            stats = {'epochs': []}
        
        stats['epochs'].append({
            'epoch': epoch + 1,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def generate_training_report(self):
        """Eğitim raporu"""
        print(f"\n{'='*60}")
        print("EĞİTİM RAPORU")
        print(f"{'='*60}")
        
        stats = self.learner.get_stats()
        
        print(f"Toplam Tarama: {stats['total_scans']}")
        print(f"Başarılı Exploitler: {stats['successful_exploits']}")
        print(f"False Positives: {stats['false_positives']}")
        print(f"Öğrenilen Payload'lar: {stats['payloads_learned']}")
        
        # Knowledge base özeti
        print(f"\nÖĞRENİLEN BİLGİLER:")
        for tech, payloads in self.learner.knowledge_base.get('successful_payloads', {}).items():
            print(f"  {tech}: {len(payloads)} payload")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_training_epoch(epochs=2)
