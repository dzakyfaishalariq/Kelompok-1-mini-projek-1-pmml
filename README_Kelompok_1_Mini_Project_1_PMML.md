# Mini Project 1 PMML — Klasifikasi Citra BreastMNIST

Proyek kelompok ini membangun dan membandingkan model *deep learning* untuk mengklasifikasikan citra ultrasound payudara ke dalam dua kelas: **benign** dan **malignant**. Proyek dikerjakan sebagai bagian dari Mini Project 1 pada mata kuliah *Practical Machine Learning (PMML)*.

> **Catatan penting:** Proyek ini merupakan eksperimen akademik, bukan alat diagnosis medis. Prediksi model tidak boleh digunakan sebagai pengganti pemeriksaan atau keputusan dokter.

## Anggota Kelompok

| Nama | ID |
|---|---|
| Dzaky Faishalariq | 25/573204/PPA/07214 |
| Made Garda Setiawan | 25/573702/PPA/07218 |
| Nur Ilham Iskandar | 25/572104/PPA/07185 |
| Taufik Nurrizqi | 25/573844/PPA/07225 |

## Latar Belakang dan Tujuan

Deteksi dini kanker payudara merupakan masalah penting dalam bidang kesehatan. Proyek ini mengeksplorasi penggunaan visi komputer dan *deep learning* untuk mengklasifikasikan citra ultrasound payudara. Selain membangun model baseline, kelompok juga menguji beberapa strategi untuk mengurangi *overfitting*.

Tujuan utama:
- Memahami karakteristik dataset BreastMNIST melalui eksplorasi data.
- Membangun model klasifikasi citra berbasis CNN.
- Membandingkan SimpleCNN, ResNet18, dan MobileNetV2.
- Menguji augmentasi data, Dropout, regularisasi L2 (*weight decay*), *learning-rate scheduler*, dan *early stopping*.
- Mengevaluasi perilaku model menggunakan kurva loss dan *confusion matrix*, dengan perhatian khusus pada *recall* kelas malignant.

## Dataset

Proyek menggunakan **BreastMNIST** dari paket [MedMNIST](https://medmnist.com/). Dataset terdiri atas citra grayscale berukuran 28 × 28 piksel dan label biner. Notebook memuat pembagian data train, validation, dan test melalui API MedMNIST.

Dataset diunduh otomatis oleh `BreastMNIST(..., download=True)` saat notebook dijalankan dan koneksi internet tersedia.

**Pemetaan label harus diperhatikan:** pada bagian evaluasi notebook, label `0` diperlakukan sebagai malignant dan label `1` sebagai benign. Pastikan pemetaan ini konsisten dengan metadata resmi dataset dan kode evaluasi sebelum menginterpretasikan hasil.

## Alur Pengerjaan

1. **Business Understanding** — mendefinisikan masalah klasifikasi citra dan kebutuhan untuk memperhatikan kesalahan *false negative* pada kelas malignant.
2. **Data Understanding** — memuat BreastMNIST, memeriksa jumlah data dan distribusi kelas, serta menampilkan contoh citra.
3. **Data Preparation** — mengubah citra menjadi tensor, melakukan normalisasi, dan menerapkan augmentasi pada data training.
4. **Modeling** — membangun dan menyiapkan tiga arsitektur:
   - **SimpleCNN** — CNN sederhana yang dibuat khusus untuk citra grayscale 28 × 28.
   - **ResNet18** — ResNet18 yang layer input-nya disesuaikan menjadi satu kanal dan output-nya menjadi satu logit untuk klasifikasi biner.
   - **MobileNetV2** — MobileNetV2 dengan layer input dan classifier yang disesuaikan untuk citra satu kanal dan klasifikasi biner.
5. **Training** — melatih model menggunakan `BCEWithLogitsLoss`, optimizer Adam, dan scheduler `ReduceLROnPlateau`.
6. **Mitigasi Overfitting** — menguji augmentasi yang lebih kuat, Dropout, regularisasi L2 melalui `weight_decay`, scheduler, dan *early stopping*.
7. **Evaluasi** — menampilkan kurva training/validation loss serta *confusion matrix* untuk membandingkan prediksi model.
8. **Audit eksperimen** — notebook juga menyediakan bagian pemeriksaan riwayat loss, ringkasan eksperimen, integritas checkpoint, dan evaluasi test set yang memerlukan artefak sumber tertentu.

## Teknologi yang Digunakan

- Python
- Jupyter Notebook
- PyTorch
- Torchvision
- MedMNIST
- NumPy dan Pandas
- Matplotlib
- scikit-learn
- torchsummary
- nbformat (untuk bagian audit notebook)

## Persiapan Lingkungan

Disarankan menggunakan Python 3.10 atau versi yang kompatibel dengan versi PyTorch dan Torchvision yang dipasang.

Buat virtual environment (opsional tetapi disarankan):

```bash
python -m venv .venv
```

Aktifkan environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Pasang dependensi:

```bash
pip install jupyter medmnist torch torchvision torchsummary numpy pandas matplotlib scikit-learn nbformat
```

Untuk instalasi PyTorch yang sesuai dengan GPU/CUDA, ikuti instruksi resmi di [pytorch.org](https://pytorch.org/get-started/locally/).

## Cara Menjalankan

1. Clone repository:

   ```bash
   git clone <URL-REPOSITORY>
   cd <NAMA-REPOSITORY>
   ```

2. Pastikan dependensi sudah terpasang.
3. Jalankan Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Buka `Kelompok_1_Mini_Project_1_PMML.ipynb`.
5. Jalankan cell dari atas ke bawah. Unduhan dataset memerlukan koneksi internet.
6. Periksa kurva loss dan hasil evaluasi yang dihasilkan oleh notebook.

### Catatan untuk bagian audit di akhir notebook

Bagian audit eksperimen merujuk pada notebook sumber lama dan file ZIP yang berisi CSV loss baseline. Bagian tersebut mencari nama berkas seperti `Mini_Projek_1_PMML_V2.ipynb` dan `Mini Project V2.zip` (atau variasi nama yang ditentukan dalam kode). Jika berkas tersebut tidak disertakan di repository, sesuaikan `ORIGINAL_NAMES`, `ZIP_NAMES`, dan lokasi sumber pada cell audit, atau jalankan bagian analisis utama tanpa cell audit tersebut.

Evaluasi test final juga memerlukan file `breastmnist.npz` dan checkpoint model yang sesuai. Jika artefak ini tidak tersedia, notebook akan melewati atau menghentikan bagian audit terkait; jangan menganggap metrik test sudah dihasilkan jika cell evaluasi belum berhasil dijalankan.

## Strategi Mengurangi Overfitting

Notebook menguji beberapa pendekatan:
- **Augmentasi data:** rotasi, horizontal flip, vertical flip, serta penyesuaian brightness/contrast pada eksperimen robust.
- **Dropout:** menonaktifkan sebagian aktivasi selama training agar model tidak terlalu bergantung pada fitur tertentu.
- **L2 regularization / weight decay:** memberi penalti pada bobot model.
- **Learning-rate scheduler:** menurunkan learning rate saat validation loss tidak membaik.
- **Early stopping:** menghentikan training jika validation loss tidak membaik setelah sejumlah epoch.

Efektivitas strategi perlu dinilai dari data validation dan test yang terpisah, bukan hanya dari training loss.

## Metrik Evaluasi

Notebook menggunakan atau membahas metrik berikut:
- **Accuracy** — proporsi prediksi yang benar.
- **Precision** — ketepatan prediksi pada kelas positif.
- **Recall/Sensitivity** — proporsi kasus positif yang berhasil dikenali.
- **F1-score** — rata-rata harmonik precision dan recall.
- **ROC-AUC** — kemampuan model membedakan kelas berdasarkan skor prediksi.
- **Confusion matrix** — rincian prediksi benar dan salah untuk setiap kelas.
- **Training/validation loss** — membantu menilai proses pembelajaran dan indikasi overfitting.

Dalam konteks skrining, *recall* untuk kelas malignant penting untuk dianalisis. Namun, semua metrik harus ditafsirkan bersama ukuran dataset, ketidakseimbangan kelas, dan keterbatasan eksperimen.

## Struktur Repository yang Disarankan

```text
.
├── README.md
├── Kelompok_1_Mini_Project_1_PMML.ipynb
├── requirements.txt              # opsional
├── data/                         # opsional; ikuti ketentuan lisensi dataset
└── outputs/                      # grafik, tabel evaluasi, dan hasil eksperimen
```

Checkpoint model (`.pth`), CSV hasil eksperimen, dan dataset tidak wajib dimasukkan ke GitHub. Jika ukurannya besar, simpan di penyimpanan yang sesuai atau gunakan Git LFS bila diperlukan. Hindari mengunggah file sensitif atau artefak yang tidak memiliki izin distribusi.

## Batasan

- Citra BreastMNIST berukuran kecil (28 × 28 piksel), sehingga hasilnya belum tentu mewakili citra klinis beresolusi tinggi.
- Hasil model dipengaruhi oleh ukuran dan distribusi dataset, augmentasi, pemilihan threshold, serta konfigurasi training.
- Perbandingan yang adil memerlukan preprocessing, pembagian data, dan prosedur evaluasi yang konsisten.
- Model ini hanya untuk pembelajaran dan penelitian akademik, bukan penggunaan klinis.

## Referensi dan Tautan

- MedMNIST: https://medmnist.com/
- MedMNIST GitHub: https://github.com/MedMNIST/MedMNIST
- PyTorch: https://pytorch.org/
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

---

Dibuat sebagai dokumentasi proyek kelompok untuk **Mini Project 1 — PMML**.
