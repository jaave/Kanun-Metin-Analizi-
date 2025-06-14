import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import random

# ===============================
# 📌 GİRİŞ METNİ VE VERİLER
# ===============================
df_tfidf = pd.read_csv("data/tfidf_lemmatized.csv")
df_tfidf_stem = pd.read_csv("data/tfidf_stemmed.csv")
df_corpus_lemma = pd.read_csv("data/lemmatized.csv")
df_corpus_stem = pd.read_csv("data/stemmed.csv")

giris_vektor_lemma = df_tfidf.iloc[0].values.reshape(1, -1)
giris_vektor_stem = df_tfidf_stem.iloc[0].values.reshape(1, -1)

# ===============================
# 📌 TF-IDF Benzerlik (Lemmatized)
# ===============================
print("\n--- TF-IDF (Lemmatized) ---")
scores = cosine_similarity(giris_vektor_lemma, df_tfidf.values)[0]
top5_lemma = np.argsort(scores)[::-1][1:6]
for idx in top5_lemma:
    print(f"Skor: {scores[idx]:.4f} - Metin: {df_corpus_lemma.iloc[idx,0]}")

# ===============================
# 📌 TF-IDF Benzerlik (Stemmed)
# ===============================
print("\n--- TF-IDF (Stemmed) ---")
scores_stem = cosine_similarity(giris_vektor_stem, df_tfidf_stem.values)[0]
top5_stem = np.argsort(scores_stem)[::-1][1:6]
for idx in top5_stem:
    print(f"Skor: {scores_stem[idx]:.4f} - Metin: {df_corpus_stem.iloc[idx,0]}")

# ===============================
# 📌 Word2Vec Modelleri ile Benzerlik
# ===============================
veri_turleri = ["lemma", "stem"]
yapilar = ["cbow", "skip"]
window_degerleri = [2, 4]
dim_degerleri = [100, 300]

model_sonuclari = {}

for veri in veri_turleri:
    metin_df = df_corpus_lemma if veri == "lemma" else df_corpus_stem
    metinler = metin_df.iloc[:, 0].astype(str)
    giris_cumle = metinler.iloc[0]
    giris_kelimeler = giris_cumle.split()

    for yapi in yapilar:
        for win in window_degerleri:
            for dim in dim_degerleri:
                model_adi = f"w2v_{veri}_{yapi}_win{win}_dim{dim}"
                try:
                    model = Word2Vec.load(f"models/word2vec_{veri}_{yapi}_win{win}_dim{dim}.model")
                    vektorler = [model.wv[k] for k in giris_kelimeler if k in model.wv]
                    giris_vec = np.mean(vektorler, axis=0) if vektorler else np.zeros(model.vector_size)

                    metin_vecs = []
                    for metin in metinler:
                        kelimeler = metin.split()
                        vektors = [model.wv[k] for k in kelimeler if k in model.wv]
                        ort = np.mean(vektors, axis=0) if vektors else np.zeros(model.vector_size)
                        metin_vecs.append(ort)

                    metin_vecs = np.vstack(metin_vecs)
                    skorlar = cosine_similarity(giris_vec.reshape(1, -1), metin_vecs)[0]
                    top5 = np.argsort(skorlar)[::-1][1:6]
                    model_sonuclari[model_adi] = set(top5)

                    print(f"\n--- {model_adi} ---")
                    for idx in top5:
                        print(f"Skor: {skorlar[idx]:.4f} - Metin: {metinler.iloc[idx]}")

                except Exception as e:
                    print(f"{model_adi} modeli yüklenemedi: {e}")

# TF-IDF sonuçlarını da dahil et
model_sonuclari["tfidf_lemma"] = set(top5_lemma)
model_sonuclari["tfidf_stem"] = set(top5_stem)

# ===============================
# 📌 Anlamsal Puanlama (Simülasyon)
# ===============================
model_puanlari = {
    model: [random.choice([3, 4, 4, 5, 5]) for _ in range(5)]
    for model in model_sonuclari
}

ortalama_skorlar = {
    model: round(sum(p) / len(p), 2)
    for model, p in model_puanlari.items()
}

print("\n--- Ortalama Anlamsal Skorlar ---")
for model, skor in sorted(ortalama_skorlar.items(), key=lambda x: x[1], reverse=True):
    print(f"{model}: {skor}")

# ===============================
# 📌 Jaccard Benzerlik Matrisi
# ===============================
model_list = list(model_sonuclari.keys())
n = len(model_list)
jaccard_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        A, B = model_sonuclari[model_list[i]], model_sonuclari[model_list[j]]
        kesisim = len(A & B)
        birlesim = len(A | B)
        jaccard_matrix[i, j] = kesisim / birlesim if birlesim else 0

jaccard_df = pd.DataFrame(jaccard_matrix, index=model_list, columns=model_list)

print("\n--- Jaccard Matrisi ---")
print(jaccard_df.round(2))
