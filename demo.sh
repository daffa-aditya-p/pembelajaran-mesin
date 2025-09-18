#!/bin/bash
# demo.sh - Script demo untuk teman pemula

echo "🔬 DEMO PREDIKSI RISIKO KANKER PAYUDARA"
echo "========================================="
echo ""

# Aktivasi environment
echo "📦 Mengaktifkan environment..."
source venv/bin/activate
echo "✅ Environment aktif!"
echo ""

echo "🤖 Informasi Model:"
python main.py --model-dir "output_final/models" --info
echo ""

echo "🧑‍⚕️ Test Sample Pasien Risiko Rendah:"
echo "----------------------------------------"
python main.py --model-dir "output_final/models" --input "sample_patient_low_risk.json"
echo ""

echo "🚨 Test Sample Pasien Risiko Tinggi:"
echo "-------------------------------------"
python main.py --model-dir "output_final/models" --input "sample_patient_high_risk.json"
echo ""

echo "📊 Batch Prediction (5 pasien):"
echo "--------------------------------"
python main.py --model-dir "output_final/models" --batch --input "sample_batch_patients.csv" --output "demo_hasil.json"
echo ""
echo "✅ Hasil batch disimpan di: demo_hasil.json"
echo ""

echo "🎉 DEMO SELESAI!"
echo "=================="
echo "Untuk mode interaktif, jalankan:"
echo "python main.py --model-dir \"output_final/models\" --interactive"
echo ""