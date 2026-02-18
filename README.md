# 🕵️‍♂️ AI-Cyber-Agent: Hybrid Pentest Framework

**AI-Cyber-Agent**, geleneksel otomasyon ile Takviyeli Öğrenme (Reinforcement Learning) zekasını birleştiren modern bir sızma testi platformudur. Sistem, hedefi analiz etmek, zafiyetleri keşfetmek ve bunları doğrulamak için hem deterministik algoritmalar hem de eğitilmiş yapay zeka modelleri kullanır.

## 🌟 Hibrit Mimari

Bu proje iki ana çalışma moduna sahiptir:

1.  **Core Agent (Deterministik):** `main.py` üzerinden çalışır. Standart bir pentest akışını (Crawl -> Tech -> CVE -> Exploit) takip eder.
2.  **RL Agent (Akıllı):** `inference_agent.py` üzerinden çalışır. Eğitilmiş bir PPO (Proximal Policy Optimization) modeli kullanarak, hedefin durumuna göre en mantıklı saldırı adımına karar verir.

## 🚀 Özellikler

- **Akıllı Süzme Sistemi:** Faz 1-7 arası profesyonel pentest metodolojisi.
- **RL Destekli Karar Mekanizması:** Stable Baselines3 tabanlı öğrenme yeteneği.
- **AI Kaynak Kod Analizi:** Ollama entegrasyonu ile kaynak kod üzerinden zafiyet tespiti.
- **Otomatik Raporlama:** Tüm bulgular `reports/` dizini altında JSON ve Markdown olarak saklanır.
- **WAF Atlatma:** Gelişmiş payload mutasyon ve encode teknikleri.

## 🛠️ Kurulum

1. **Gereksinimler:** Python 3.9+, [Ollama](https://ollama.ai/) ve Docker (Eğitim hedefleri için).
2. **Kurulum:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
3. **API Ayarları:** `.env` dosyasını oluşturup `NVD_API_KEY` değerini ekleyin.

## 💻 Kullanım

### A. Standart Tarama (Hızlı & Sabit Akış)
```bash
python system/PentestAiAgent/main.py --target http://hedef-site.com
```

### B. Akıllı Tarama (Trained RL Model)
```bash
python inference_agent.py --target http://hedef-site.com
```

### C. Eğitim (Modeli Kendin Geliştir)
Agent'ın zekasını artırmak için kendi yerel ortamınızda eğitim gerçekleştirebilirsiniz.

#### 1. Lab Ortamının Hazırlanması (Docker)
Eğitim için gerçek zafiyetli hedeflere ihtiyaç vardır. Proje kökündeki `docker-config.yaml` dosyasını kullanarak lab ortamını ayağa kaldırın:
```bash
docker compose -f docker-config.yaml up -d
```
Bu komut şu hedefleri başlatır:
- **DVWA:** SQLi ve Brute Force testleri için.
- **Juice Shop:** Modern OWASP zafiyetleri için.
- **WebGoat:** Derinlemesine Java zafiyet analizi için.

#### 2. Eğitim Pipeline'ını Başlatma
Lab ortamı hazır olduğunda, Reinforcement Learning (RL) sürecini başlatın:
```bash
python system/Train/TrainingPipeline.py
```
- Eğitim sırasında model `system/Train/logs/` dizinine periyodik olarak kaydedilir.
- Süreci takip etmek için TensorBoard kullanabilirsiniz: `tensorboard --logdir=system/Train/logs/`

#### 3. Yeni Modeli Kullanma
Eğitim bittiğinde oluşan `.zip` dosyasını `inference_agent.py` üzerinden şu şekilde koşturabilirsiniz:
```bash
python system/Train/inference_agent.py --model "yeni_model_yolu.zip" --target http://localhost:8080
```

## 📊 Raporlama
Tüm tarama sonuçları otomatik olarak projenin kök dizinindeki veya ilgili modül altındaki `reports/` klasörüne kaydedilir:
- `pentest_report_TARIH.json`: Detaylı veri analizi için.
- `pentest_report_TARIH.md`: Profesyonel sızma testi raporu.

## ⚖️ Yasal Uyarı ve Etik Kullanım

> [!WARNING]
> **BU ARAÇ SADECE EĞİTİM VE YASAL TESTLER İÇİNDİR.**
> 1. Bu aracı sızma testi izniniz olmayan hiçbir sistemde kullanmayın.
> 2. İzinsiz kullanım yerel ve uluslararası siber suç kanunlarına göre suçtur.
> 3. Geliştirici, aracın kötüye kullanımından veya sistemler üzerinde yaratabileceği olası zararlardan sorumlu tutulamaz.
> 4. Kullanıcı, tüm test işlemlerinin yasal sorumluluğunu peşinen kabul eder.

---
*PentestAI - Yapay Zeka ile Daha Güvenli Bir Web.*
