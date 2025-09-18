# 🚀 Quick Clone & Run Guide

## 📥 Clone Repository

```bash
git clone https://github.com/daffa-aditya-p/pembelajaran-mesin.git
cd pembelajaran-mesin
```

## ⚡ Quick Start (Super Gampang!)

### 1️⃣ Setup Environment
```bash
# Otomatis install semua dependencies
source setup_env.sh

# Aktivasi environment
source venv/bin/activate
```

### 2️⃣ Demo Lengkap (Sekali Klik!)
```bash
# Jalankan demo otomatis - lihat semua fitur
./demo.sh
```

### 3️⃣ Test Manual
```bash
# Info model
python main.py --model-dir "output_final/models" --info

# Test risiko rendah
python main.py --model-dir "output_final/models" --input "sample_patient_low_risk.json"

# Test risiko tinggi  
python main.py --model-dir "output_final/models" --input "sample_patient_high_risk.json"

# Batch prediction (5 pasien)
python main.py --model-dir "output_final/models" --batch --input "sample_batch_patients.csv" --output "hasil.json"

# Mode interaktif
python main.py --model-dir "output_final/models" --interactive
```

## 📚 Documentation

- **README.md** - Dokumentasi lengkap dan profesional
- **README_SIMPLE.md** - Panduan untuk pemula Python
- **WORKSPACE_STRUCTURE.md** - Struktur project

## 🤖 Model Ready!

✅ **RandomForest** dengan 100% accuracy  
✅ **Trained model** siap pakai (no training needed!)  
✅ **Sample data** untuk testing instant  
✅ **Interactive mode** untuk input manual  
✅ **Batch prediction** untuk banyak pasien  

## 🎯 One-Liner untuk Teman Pemula

```bash
git clone https://github.com/daffa-aditya-p/pembelajaran-mesin.git && cd pembelajaran-mesin && source setup_env.sh && source venv/bin/activate && ./demo.sh
```

**Tinggal copy-paste command di atas, everything works!** 🚀✨