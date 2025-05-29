import pdfplumber
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from nltk.corpus import stopwords
from trnlp import TrnlpWord
from gensim.models import Word2Vec

def indir_nltk_kaynaklari():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print(" stopwords indiriliyor...")
        nltk.download('stopwords')

stop_words = set(stopwords.words("turkish"))

lemmatizer = TrnlpWord()

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def clean_and_preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zçğıöşü0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatized = []
    for token in tokens:
        lemmatizer.setword(token)
        lemma = lemmatizer.get_stem
        if lemma:
            lemmatized.append(lemma)
        else:
            lemmatized.append(token)
    return " ".join(lemmatized)

def extract_lemmatized_list(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zçğıöşü0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatized = []
    for token in tokens:
        lemmatizer.setword(token)
        lemma = lemmatizer.get_stem
        if lemma:
            lemmatized.append(lemma)
        else:
            lemmatized.append(token)
    return lemmatized

def save_csv(kelimeler, dosya_adi):
    df = pd.DataFrame(Counter(kelimeler).items(), columns=["Kelime", "Frekans"])
    df = df.sort_values(by="Frekans", ascending=False)
    df.to_csv(dosya_adi, index=False, encoding="utf-8-sig")
    print(f"[✓] Kaydedildi: {dosya_adi}")
    return df

def zipf_grafik(df, title, filename):
    ranks = np.arange(1, len(df) + 1)
    freqs = df["Frekans"].values
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(ranks), np.log(freqs), marker='o', linestyle='-', markersize=3)
    plt.xlabel("log(Sıralama - Rank)")
    plt.ylabel("log(Frekans)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[✓] Zipf grafiği kaydedildi: {filename}")
    print(f" {title} - Toplam kelime: {sum(freqs)} | Farklı kelime: {len(df)}")

def parse_maddeler(text):
    parts = re.split(r'\bMADDE\s*\d+\b', text, flags=re.IGNORECASE)
    headers = re.findall(r'\bMADDE\s*\d+\b', text, flags=re.IGNORECASE)
    if not headers or len(parts) <= 1:
        return []
    maddeler = []
    for i in range(len(headers)):
        madde_no = headers[i].strip().upper()
        madde_icerik = parts[i + 1].strip()
        temizlenmis = clean_and_preprocess(madde_icerik)
        maddeler.append((madde_no, temizlenmis))
    return maddeler

def tfidf_karsilastir(eski_maddeler, yeni_maddeler, esik=0.75):
    eski_met = [m[1] for m in eski_maddeler]
    yeni_met = [m[1] for m in yeni_maddeler]
    tfidf = TfidfVectorizer()
    matris = tfidf.fit_transform(eski_met + yeni_met)
    eski_vec = matris[:len(eski_met)]
    yeni_vec = matris[len(eski_met):]
    similarity_matrix = cosine_similarity(yeni_vec, eski_vec)

    sonuc = []
    for i, yeni in enumerate(yeni_maddeler):
        max_score = similarity_matrix[i].max()
        j = similarity_matrix[i].argmax()
        eski_no, _ = eski_maddeler[j]
        durum = "Benzer" if max_score >= esik else "Yeni Madde"
        sonuc.append({
            "Yeni Madde No": yeni[0],
            "Yeni Madde İçeriği": yeni[1],
            "Eşleşen Eski Madde": eski_no,
            "Benzerlik": round(max_score, 3),
            "Durum": durum
        })

    eslesen = {similarity_matrix[i].argmax() for i in range(len(yeni_maddeler)) if similarity_matrix[i].max() >= esik}
    for k, (eski_no, eski_ic) in enumerate(eski_maddeler):
        if k not in eslesen:
            sonuc.append({
                "Yeni Madde No": "-",
                "Yeni Madde İçeriği": "-",
                "Eşleşen Eski Madde": eski_no,
                "Benzerlik": 0.0,
                "Durum": "Çıkarılmış Madde"
            })

    return pd.DataFrame(sonuc)

def tfidf_vektorleştir(kaynak_dosya, hedef_dosya):
    df = pd.read_csv(kaynak_dosya)
    metinler = df["Kelime"].repeat(df["Frekans"]).tolist()
    belge = " ".join(metinler)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([belge])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.to_csv(hedef_dosya, index=False, encoding="utf-8-sig")
    print(f" TF-IDF verisi kaydedildi: {hedef_dosya}")

def word2vec_modellerini_egit(csv_path, kayit_klasoru, ornek_kelimeler=["hukuk"]):
    df = pd.read_csv(csv_path)
    kelimeler = df["Kelime"].repeat(df["Frekans"]).tolist()
    cümleler = [kelimeler[i:i+30] for i in range(0, len(kelimeler), 30)]

    parameters = [
        {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
        {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
        {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
        {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    ]

    for p in parameters:
        sg = 0 if p['model_type'] == 'cbow' else 1
        model = Word2Vec(
            sentences=cümleler,
            vector_size=p['vector_size'],
            window=p['window'],
            sg=sg,
            min_count=1,
            workers=4,
            epochs=50
        )

        model_adi = f"word2vec_lemma_{p['model_type']}_win{p['window']}_dim{p['vector_size']}.model"
        yol = os.path.join(kayit_klasoru, model_adi)
        model.save(yol)
        print(f" Model kaydedildi: {yol}")

        for kelime in ornek_kelimeler:
            if kelime in model.wv:
                benzerler = model.wv.most_similar(kelime, topn=5)
                print(f" {kelime} kelimesine en yakın kelimeler ({model_adi}):")
                for kel, sk in benzerler:
                    print(f"    - {kel} ({sk:.3f})")
            else:
                print(f"[!] '{kelime}' kelimesi modelde bulunamadı. ({model_adi})")

def main():
    indir_nltk_kaynaklari()

    eski_pdf = "data/eski.pdf"
    yeni_pdf = "data/yeni.pdf"

    if not os.path.exists(eski_pdf) or not os.path.exists(yeni_pdf):
        print("[!] PDF dosyaları bulunamadı.")
        return

    print("[1] PDF'ten metin çıkarılıyor...")
    eski_text = extract_text_from_pdf(eski_pdf)
    yeni_text = extract_text_from_pdf(yeni_pdf)

    print("[2] Lemmatized kelime listeleri oluşturuluyor...")
    eski_lemmas = extract_lemmatized_list(eski_text)
    yeni_lemmas = extract_lemmatized_list(yeni_text)

    df_eski = save_csv(eski_lemmas, "data/eski_lemmatized.csv")
    df_yeni = save_csv(yeni_lemmas, "data/yeni_lemmatized.csv")

    print("[3] Zipf grafik çizimleri yapılıyor...")
    zipf_grafik(df_eski, "Zipf - Eski Kanun (Lemmatized)", "output/zipf_eski_lemma.png")
    zipf_grafik(df_yeni, "Zipf - Yeni Kanun (Lemmatized)", "output/zipf_yeni_lemma.png")

    print("[4] Madde ayrıştırılıyor + benzerlik analizine hazırlanıyor...")
    eski_maddeler = parse_maddeler(eski_text)
    yeni_maddeler = parse_maddeler(yeni_text)
    print(f" {len(eski_maddeler)} eski, {len(yeni_maddeler)} yeni madde bulundu.")

    df = tfidf_karsilastir(eski_maddeler, yeni_maddeler)
    df.to_csv("output/results_lemma.csv", index=False, encoding="utf-8-sig")
    print("Benzerlik sonuçları kaydedildi: output/results_lemma.csv")

    print("[5] TF-IDF vektörleştirme başlatılıyor...")
    tfidf_vektorleştir("data/eski_lemmatized.csv", "data/tfidf_lemmatized.csv")

    print("[6] Word2Vec modelleri eğitiliyor...")
    word2vec_modellerini_egit("data/eski_lemmatized.csv", "models")

if __name__ == "__main__":
    main()
