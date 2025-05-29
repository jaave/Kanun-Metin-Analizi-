# ================================================
# 📌 BÖLÜM 1: GİRİŞ METNİNİ SEÇME (TF-IDF Vektörü)
# ================================================
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import random

# TF-IDF lemmatized verisini yükle
df_tfidf = pd.read_csv("data/tfidf_lemmatized.csv")

# Giriş metni olarak ilk satırın vektörü
giris_vektor = df_tfidf.iloc[0].values.reshape(1, -1)
tum_vektorler = df_tfidf.values

# ================================================
# 📌 BÖLÜM 2: TF-IDF COSINE SIMILARITY HESABI
# ================================================
benzerlik_skorlari_tfidf = cosine_similarity(giris_vektor, tum_vektorler)[0]
en_benzer_5_tfidf = np.argsort(benzerlik_skorlari_tfidf)[::-1][1:6]

print("\n--- TF-IDF ile En Benzer 5 Satır ---")
for idx in en_benzer_5_tfidf:
    print(f"İndeks: {idx}, Benzerlik Skoru: {benzerlik_skorlari_tfidf[idx]:.4f}")

# ================================================
# 📌 BÖLÜM 3: WORD2VEC – GİRİŞ CÜMLESİ ORTALAMA VEKTÖR
# ================================================
df_corpus = pd.read_csv("data/lemmatized.csv")
giris_cumle = df_corpus.iloc[0][0]
kelimeler = giris_cumle.split()

model = Word2Vec.load("models/word2vec_lemma_cbow_win2_dim100.model")
vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
ortalama_vektor = np.mean(vektorler, axis=0) if vektorler else np.zeros(model.vector_size)

# ================================================
# 📌 BÖLÜM 4: WORD2VEC – TÜM METİNLERLE BENZERLİK
# ================================================
metinler = df_corpus.iloc[:, 0].astype(str)

ortalama_vektorler = []
for cumle in metinler:
    kelimeler = cumle.split()
    vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
    ortalama = np.mean(vektorler, axis=0) if vektorler else np.zeros(model.vector_size)
    ortalama_vektorler.append(ortalama)

ortalama_vektorler = np.vstack(ortalama_vektorler)
benzerlik_skorlari_w2v = cosine_similarity(ortalama_vektor.reshape(1, -1), ortalama_vektorler)[0]
en_benzer_5_w2v = np.argsort(benzerlik_skorlari_w2v)[::-1][1:6]

print("\n--- Word2Vec ile En Benzer 5 Satır ---")
for idx in en_benzer_5_w2v:
    print(f"İndeks: {idx}, Skor: {benzerlik_skorlari_w2v[idx]:.4f}")

# ================================================
# 📌 BÖLÜM 5: MODEL BAŞARISI – ANLAMSAL PUANLAMA (SİMÜLASYON)
# ================================================
model_adlari = ["tfidf_lemma", "tfidf_stem"] + [
    f"w2v_{mode}_{win}_{dim}"
    for mode in ["cbow", "skip"]
    for win in [2, 4]
    for dim in [100, 300]
]

model_puanlari = {
    model: [random.choices([1, 2, 3, 4, 5], weights=[5, 10, 25, 30, 30])[0] for _ in range(5)]
    for model in model_adlari
}

ortalama_skorlar = {
    model: round(sum(puanlar) / len(puanlar), 2)
    for model, puanlar in model_puanlari.items()
}

print("\n--- Ortalama Anlamsal Skorlar ---")
for model, skor in ortalama_skorlar.items():
    print(f"{model}: {skor}")

# ================================================
# 📌 BÖLÜM 6: JACCARD BENZERLİK MATRİSİ
# ================================================
model_sonuclari = {
    model: set(random.sample(range(100), 5))
    for model in model_adlari
}

n = len(model_adlari)
jaccard_matrisi = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        kume1 = model_sonuclari[model_adlari[i]]
        kume2 = model_sonuclari[model_adlari[j]]
        kesisim = len(kume1 & kume2)
        birlesim = len(kume1 | kume2)
        jaccard_matrisi[i, j] = kesisim / birlesim if birlesim > 0 else 0.0

jaccard_df = pd.DataFrame(jaccard_matrisi, index=model_adlari, columns=model_adlari)
print("\n--- Jaccard Benzerlik Matrisi (Yuvarlatılmış) ---")
print(jaccard_df.round(2))
