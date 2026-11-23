# Klasifikasi Emosi pada Ulasan Pelanggan Menggunakan RNN

## Deskripsi Proyek
Proyek ini bertujuan untuk mengklasifikasikan emosi pada teks ulasan pelanggan berbahasa Indonesia menggunakan metode **Recurrent Neural Network (RNN)**. Model dikembangkan menggunakan arsitektur **Simple RNN** dengan bantuan *pretrained word embedding* **FastText Bahasa Indonesia** (dimensi vektor 300). Model dilatih untuk mengenali pola dan konteks kata dalam ulasan pelanggan, kemudian mengelompokkannya ke dalam lima kategori emosi: `anger`, `fear`, `happy`, `love`, dan `sadness`.

Proyek ini mencakup tahapan pengolahan data teks, tokenisasi, representasi kata menggunakan FastText, pembangunan dan pelatihan model, evaluasi performa, hingga pengujian prediksi terhadap ulasan baru.

## Tujuan
- Mempelajari penerapan *Deep Learning* untuk klasifikasi teks.
- Menerapkan *preprocessing* pada data ulasan pelanggan berbahasa Indonesia.
- Memahami penggunaan *pretrained word embedding* FastText.
- Membangun model klasifikasi teks menggunakan arsitektur Simple RNN.
- Mengatasi ketidakseimbangan jumlah data pada setiap kategori emosi menggunakan *Class Weight*.
- Mengevaluasi performa model menggunakan metrik klasifikasi standar.
- Menguji kemampuan model dalam memprediksi emosi dari ulasan baru.

## Dataset
Dataset yang digunakan adalah **PRDECT-ID** (*Product Review Emotion Classification Dataset - Indonesian*), yang berisi ulasan pelanggan produk *e-commerce* berbahasa Indonesia beserta label emosinya.
- **Jumlah data awal:** 5.400 baris
- **Jumlah data setelah pembersihan:** 5.330 data
- **Kolom teks:** `Customer Review`
- **Kolom target:** `Emotion`

### Kategori Emosi & Distribusi Data
| Kategori | Keterangan | Jumlah Data |
| :--- | :--- | :---: |
| **Happy** | Senang | 1.754 |
| **Sadness** | Sedih | 1.185 |
| **Fear** | Takut | 907 |
| **Love** | Suka atau cinta | 802 |
| **Anger** | Marah | 682 |
| **Total** | | **5.330** |

## Metode dan Teknologi
- **Python & Google Colab:** Bahasa pemrograman utama dan lingkungan pelatihan model.
- **TensorFlow & Keras:** Pembangunan dan pelatihan arsitektur Simple RNN.
- **FastText (300d):** *Pretrained word embedding* Bahasa Indonesia.
- **Pandas & NumPy:** Pengolahan dan manipulasi data tabular.
- **Scikit-learn:** *Label encoding*, pembagian data (*stratified split*), *class weighting*, dan metrik evaluasi.
- **Matplotlib & Seaborn:** Visualisasi grafik pelatihan dan *confusion matrix*.

## Tahapan Pengerjaan
1. **Memuat Dataset:** Memuat dataset PRDECT-ID dalam format CSV dan mengambil kolom `Customer Review` serta `Emotion`.
2. **Exploratory Data Analysis (EDA):** Menganalisis distribusi kategori emosi dan karakteristik panjang teks ulasan.
3. **Preprocessing Teks:** Melakukan *case folding*, penghapusan URL, *mention*, tanda baca, karakter khusus, dan spasi berlebih, dengan tetap mempertahankan kata negasi (*tidak*, *bukan*, *belum*).
4. **Encoding Label & Pembagian Data:** Mengubah label emosi ke bentuk numerik menggunakan `LabelEncoder` serta membagi dataset secara *stratified* menjadi **70% Training** (3.731 data), **15% Validation** (799 data), dan **15% Testing** (800 data).
5. **Tokenisasi & Padding:** Mengubah teks menjadi urutan indeks menggunakan `Tokenizer` (maksimal 30.000 kata dan panjang *sequence* 50 token).
6. **Penanganan Ketidakseimbangan Kelas:** Menerapkan *Class Weight* agar model mampu mempelajari kelas minoritas secara proporsional.
7. **Representasi Kata (FastText):** Memetakan kosakata ke vektor *pretrained embedding* FastText Bahasa Indonesia berdimensi 300.
8. **Arsitektur Model Simple RNN:** Menyusun lapisan *Embedding Layer*, `SpatialDropout1D`, `SimpleRNN` (64 unit), gabungan `GlobalAveragePooling1D` dan `GlobalMaxPooling1D` (`Concatenate`), `Dropout`, `Dense` (64 unit), serta *Output Dense Layer* dengan aktivasi `Softmax`.
9. **Training Model:** Melatih model hingga maksimal 30 *epoch* (`batch_size=32`, optimizer `Adam` dengan *learning rate* `0.001`) menggunakan *callback* `EarlyStopping` dan `ReduceLROnPlateau`.
10. **Evaluasi Model:** Mengukur performa pada data uji menggunakan *Accuracy*, *Macro Precision*, *Macro Recall*, *Macro F1-Score*, serta *Confusion Matrix*.
11. **Pengujian Prediksi:** Menguji model menggunakan input kalimat ulasan baru beserta skor *confidence*-nya.

## Hasil Evaluasi
Pengujian dilakukan pada **800 data testing** dengan ringkasan metrik berikut:

| Metrik | Hasil |
| :--- | :---: |
| **Accuracy** | 58,25% |
| **Macro Precision** | 54,88% |
| **Macro Recall** | 53,57% |
| **Macro F1-Score** | 54,00% |

### Classification Report per Kategori
| Kategori | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Anger** | 0,42 | 0,43 | 0,43 |
| **Fear** | 0,37 | 0,42 | 0,39 |
| **Happy** | 0,73 | 0,79 | 0,76 |
| **Love** | 0,60 | 0,48 | 0,53 |
| **Sadness** | 0,63 | 0,56 | 0,59 |

Model menunjukkan performa terbaik pada kategori **happy** (`F1-Score: 0,76`), sedangkan kategori **fear** masih menjadi tantangan utama karena kemiripan konteks kata dengan kategori emosi negatif lainnya.

## Contoh Pengujian Prediksi
- **Input Ulasan:** *"Barangnya bagus sekali, saya sangat suka dan puas dengan pembelian ini."*
- **Prediksi Emosi:** `love`
- **Confidence:** `72,85%`

## Kesimpulan
Model **Simple RNN** yang dikombinasikan dengan *pretrained embedding* **FastText Bahasa Indonesia** dan teknik *Class Weight* berhasil diterapkan untuk mengklasifikasikan lima kategori emosi pada ulasan pelanggan PRDECT-ID dengan **Accuracy 58,25%** dan **Macro F1-Score 54,00%**. Pengembangan selanjutnya dapat difokuskan pada eksplorasi arsitektur *Bidirectional RNN/LSTM/GRU* serta analisis kesalahan prediksi antar kelas emosi.

---
**Author:** Bilqizah Rahma Ilayya Syahdewi  
- **Sumber Dataset:** [PRDECT-ID Dataset](https://www.kaggle.com/datasets/jocelyndumlao/prdect-id-indonesian-emotion-classification) 
