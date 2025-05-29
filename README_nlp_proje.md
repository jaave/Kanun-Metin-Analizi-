
# ⚖️ Kanun Metin Analizi ile Metin Benzerliği (TF-IDF & Word2Vec)

Bu proje, Türk Ceza Kanunu’nun eski ve yeni versiyonlarını kullanarak doğal dil işleme (NLP) teknikleriyle metinler arası benzerlik analizi yapmaktadır. Bu analizde hem TF-IDF hem de Word2Vec modelleri kullanılmakta ve model çıktıları karşılaştırmalı olarak değerlendirilmektedir.

---

## 🔍 Proje Aşamaları

### 1. Veri Ön İşleme (Ödev 1’den devralınan)
- `lemmatized.csv`, `stemmed.csv` veri dosyaları hazır.
- `tfidf_lemmatized.csv`, `tfidf_stemmed.csv` matrisleri oluşturulmuş.
- 16 farklı Word2Vec modeli (`CBOW`, `Skip-Gram`, `window=2/4`, `dim=100/300`) eğitilmiş.

### 2. Benzerlik Analizi (Bu repo)
- Giriş metni (örnek: ilk satır) seçilir.
- Giriş metni ile tüm metinler TF-IDF ve Word2Vec kullanılarak karşılaştırılır.
- En benzer 5 metin çıkarılır.

---

## 📂 Klasör Yapısı

proje/
├── data/
│   ├── lemmatized.csv
│   ├── tfidf_lemmatized.csv
├── models/
│   ├── word2vec_lemma_cbow_win2_dim100.model
│   └── ...
├── scripts/
│   ├── benzerlik_analizi.py
├── output/
│   └── skorlar_ve_jaccard.csv (opsiyonel)
└── README.md

---

## ⚙️ Kullanım

1. Ortamı kur:
pip install pandas numpy scikit-learn gensim

2. Python scripti çalıştır:
python scripts/benzerlik_analizi.py

3. Çıktılar:
- Konsolda: en benzer metinler, skorlar, Jaccard matrisi
- (İsteğe bağlı) skor ve matrisi CSV olarak `output/` klasörüne yazabilirsin

---

## 📈 Modellerin Karşılaştırılması

| Model Adı         | Ortalama Anlamsal Skor |
|-------------------|------------------------|
| tfidf_lemma       | 4.0                    |
| w2v_skip_4_100    | 4.4                    |
| ...               | ...                    |

---

## 📊 Değerlendirme Ölçütleri

- **Anlamsal Puanlama (1–5):** Giriş metni ile olan anlam yakınlığı.
- **Jaccard Benzerlik Matrisi:** İlk 5 sonucu benzer modellerin tespiti.
- **Model Etkisi:** CBOW vs Skip-gram, pencere genişliği ve vektör boyutu etkileri incelenmiştir.

---

## 👨‍💻 Geliştirici

Güner Bektaş – NLP Final Projesi (2025)
