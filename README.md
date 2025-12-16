📊 Dashboard Prediksi Suhu BMKG - Regresi Linear & Polynomial

📋 Deskripsi Proyek
Dashboard interaktif untuk memprediksi suhu udara berdasarkan kelembaban relatif menggunakan metode Least Squares Method (LSM). Proyek ini mengimplementasikan regresi linear dan polynomial derajat 2 dengan visualisasi data real-time dari BMKG Stasiun Raja Haji Fisabilillah.

🎯 Fitur Utama
📈 Regresi Linear - Model prediksi T = m × RH + c
📊 Regresi Polynomial - Model prediksi T = aRH² + bRH + c
📉 Residual Plot - Analisis error dengan garis nol y=0
🔢 Input Real-time - Slider kelembaban dengan update instan
📋 Tabel Data - Sample data dengan error per observasi
📥 Export CSV - Download semua prediksi dengan metadata
📊 Statistik Lengkap - R², MAE, MSE, RMSE, Max Error

🚀 Cara Menjalankan
Metode 1: Langsung di Browser
Download semua file: tampilan.html, script.js, data.csv
Simpan dalam satu folder yang sama
Buka tampilan.html di browser (Chrome/Firefox/Edge)

Metode 2: Menggunakan Live Server
# Install live-server (jika belum ada)
npm install -g live-server
# Jalankan di folder proyek
live-server

📁 Struktur File
dashboard-bmkg/
├── tampilan.html         # File HTML utama
├── script.js             # Logika JavaScript & LSM
├── data.csv              # Dataset BMKG (30 observasi)
└── README.md             # Dokumentasi ini

🔧 Teknologi yang Digunakan
HTML5 & CSS3 - Struktur dan styling dashboard
JavaScript ES6 - Logika aplikasi dan kalkulasi LSM
Chart.js v3.9.1 - Visualisasi grafik interaktif
Pure Math - Implementasi algoritma Least Squares Method

📊 Dataset
Sumber: BMKG Stasiun Raja Haji Fisabilillah
Periode: 1-30 November 2025
Jumlah Data: 30 observasi
Variabel:
    > Kelembaban Relatif (%)
    > Suhu Udara (°C)

📈 Model Matematika
Regresi Linear (LSM)
T = m × RH + c
dimana:
m = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²)
c = (Σy - mΣx) / n
Regresi Polynomial Degree 2 (LSM)
T = aRH² + bRH + c
Sistem persamaan normal:
[ n    Σx    Σx² ] [c]   [Σy]
[ Σx   Σx²   Σx³ ] [b] = [Σxy]
[ Σx²  Σx³   Σx⁴ ] [a]   [Σx²y]

🎮 Panduan Penggunaan
1. Input Data
Gunakan slider kelembaban (70-95%)
Prediksi akan update otomatis
Lihat hasil di panel kiri

2. Interpretasi Grafik
● Titik biru/merah: Data aktual BMKG
— Garis biru/ungu: Garis/kurva regresi
★ Bintang oranye: Prediksi saat ini
--- Garis hitam putus-putus: Garis nol (residual plot)

3. Analisis Statistik
R² (R-squared): Seberapa baik model menjelaskan variasi data
MAE: Rata-rata error absolut
RMSE: Akar rata-rata kuadrat error
Max Error: Error terbesar dalam dataset

4. Klasifikasi Status
✅ AKURAT: Error < ±0.5°C
⚠️ NETRAL: Error ±0.5-1.0°C
🔴 PERHATIAN: Error > ±1.0°C

📥 Export Data
Klik tombol "Download Prediksi CSV" untuk export:
Semua data observasi
Prediksi linear & polynomial
Error per model
Status akurasi
Metadata model

🐛 Troubleshooting
Masalah Umum:
Grafik tidak muncul: Pastikan file data.csv ada di folder yang sama
Slider tidak bekerja: Refresh halaman (F5)
Data tidak load: Cek koneksi internet untuk load Chart.js

Browser Support:
✅ Google Chrome (disarankan)
✅ Mozilla Firefox
✅ Microsoft Edge
✅ Safari

👥 Tim Pengembang
Kelompok 4 - Penerapan PDE dan LSM
Rauf Hidayat            2401020057
Jova Rifana             2401020050
Januar Suyasmin Saputra 2401020046
Muhammad Adib Haryadi   2401020076

📅 Informasi Proyek
Versi: 1.0
Deadline: 16 Desember 2025
Mata Kuliah: Penerapan PDE dan LSM
Institusi: [Nama Universitas]

📄 Lisensi
Proyek ini dibuat untuk tujuan akademik. Silakan gunakan dan modifikasi dengan mencantumkan sumber.

🔗 Referensi
BMKG - Badan Meteorologi, Klimatologi, dan Geofisika
Chart.js Documentation
Least Squares Method - Numerical Analysis