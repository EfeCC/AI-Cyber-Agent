import json
import os
from datetime import datetime

class ReinforcementLearner:
    def __init__(self):
        self.knowledge_base = self.load_knowledge()
        self.stats = {
            'total_scans': 0,
            'successful_exploits': 0,
            'false_positives': 0,
            'payloads_learned': 0
        }
    
    def load_knowledge(self):
        """Öğrenilmiş bilgileri yükle"""
        if os.path.exists('knowledge_base.json'):
            with open('knowledge_base.json', 'r') as f:
                return json.load(f)
        return {
            'successful_payloads': {},
            'failed_payloads': {},
            'target_fingerprints': {}
        }
    
    def calculate_reward(self, action_result):
        """Reward hesapla"""
        rewards = {
            'exploit_success': +10,      # Başarılı exploit
            'exploit_verified': +20,     # Doğrulanmış zafiyet
            'false_positive': -5,        # Yanlış alarm
            'time_efficient': +2,        # Hızlı tespit
            'new_technique': +15,        # Yeni teknik keşfi
            'waf_bypass': +25            # WAF bypass başarısı
        }
        
        total_reward = 0
        
        # Başarılı exploit
        if action_result.get('exploit_verified'):
            total_reward += rewards['exploit_success']
            
            # İlk kez bu payload başarılı olduysa
            payload = action_result.get('payload')
            if payload not in self.knowledge_base['successful_payloads']:
                total_reward += rewards['new_technique']
        
        # Yanlış pozitif
        if action_result.get('false_positive'):
            total_reward += rewards['false_positive']
        
        # Süre performansı
        if action_result.get('execution_time', 100) < 10:
            total_reward += rewards['time_efficient']
        
        return total_reward
    
    def learn_from_exploit(self, exploit_data):
        """Başarılı exploit'ten öğren"""
        payload = exploit_data.get('payload')
        target_tech = exploit_data.get('technology', 'unknown')
        
        if payload:
            # Başarılı payload'ı kaydet
            if target_tech not in self.knowledge_base['successful_payloads']:
                self.knowledge_base['successful_payloads'][target_tech] = []
            
            payload_info = {
                'payload': payload,
                'success_count': 1,
                'last_success': datetime.now().isoformat(),
                'target_pattern': exploit_data.get('target_pattern'),
                'bypass_technique': exploit_data.get('bypass_technique')
            }
            
            # Eğer zaten varsa success_count artır
            existing = next((p for p in self.knowledge_base['successful_payloads'][target_tech] 
                           if p['payload'] == payload), None)
            
            if existing:
                existing['success_count'] += 1
                existing['last_success'] = datetime.now().isoformat()
            else:
                self.knowledge_base['successful_payloads'][target_tech].append(payload_info)
                self.stats['payloads_learned'] += 1
        
        self.save_knowledge()
    
    def learn_from_failure(self, failure_data):
        """Başarısız denemeden öğren"""
        payload = failure_data.get('payload')
        target_tech = failure_data.get('technology', 'unknown')
        
        if payload:
            if target_tech not in self.knowledge_base['failed_payloads']:
                self.knowledge_base['failed_payloads'][target_tech] = []
            
            # Başarısız payload'ı kaydet (bir daha denememek için)
            if payload not in self.knowledge_base['failed_payloads'][target_tech]:
                self.knowledge_base['failed_payloads'][target_tech].append(payload)
        
        self.save_knowledge()
    
    def get_best_payloads(self, technology, limit=5):
        """En başarılı payload'ları getir"""
        if technology not in self.knowledge_base['successful_payloads']:
            return []
        
        # Success_count'a göre sırala
        payloads = sorted(
            self.knowledge_base['successful_payloads'][technology],
            key=lambda x: x['success_count'],
            reverse=True
        )
        
        return payloads[:limit]
    
    def should_try_payload(self, payload, technology):
        """Bu payload'u denemeli miyiz?"""
        # Daha önce başarısız olduysa atla
        failed_list = self.knowledge_base['failed_payloads'].get(technology, [])
        if payload in failed_list:
            return False
        
        return True
    
    def save_knowledge(self):
        """Öğrenilenleri kaydet"""
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
    
    def get_stats(self):
        """Öğrenme istatistikleri"""
        return self.stats
