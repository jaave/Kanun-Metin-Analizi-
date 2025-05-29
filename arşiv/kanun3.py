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
from nltk.stem import PorterStemmer

# Sadece stopwords indiriyoruz
def indir_nltk_kaynaklari():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print(" stopwords indiriliyor...")
        nltk.download('stopwords')

stop_words = set(stopwords.words("turkish"))
stemmer = PorterStemmer()

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

    #  punkt/tokenizer kullanmadan kelimelere ayır
    tokens = re.findall(r'\b\w+\b', text)

    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    stems = [stemmer.stem(w) for w in tokens]

    return " ".join(stems)

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

def zipf_grafigi_olustur(text, title="Zipf Yasası Grafiği", filename="zipf_grafigi.png"):
    kelimeler = re.findall(r'\b\w+\b', text.lower())
    sayac = Counter(kelimeler)
    frekanslar = sorted(sayac.values(), reverse=True)
    ranks = range(1, len(frekanslar) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(np.log(ranks), np.log(frekanslar), marker='o', linestyle='-', markersize=3)
    plt.xlabel("log(Sıralama - Rank)")
    plt.ylabel("log(Frekans)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[✓] Zipf grafiği oluşturuldu: {filename}")

    print(f"\n[Zipf Yorum]")
    print(f"Toplam kelime: {len(kelimeler)} | Farklı kelime sayısı: {len(sayac)}")
    if len(sayac) < 300:
        print(" Veri seti Zipf eğrisi için küçük olabilir.")
    else:
        print(" Veri seti Zipf analizi için yeterlidir.")

def main():
    indir_nltk_kaynaklari()

    eski_pdf = "eski.pdf"
    yeni_pdf = "yeni.pdf"

    if not os.path.exists(eski_pdf) or not os.path.exists(yeni_pdf):
        print(" PDF dosyaları klasörde bulunamadı.")
        return

    print(" PDF'ten metin çıkarılıyor...")
    eski_text = extract_text_from_pdf(eski_pdf)
    yeni_text = extract_text_from_pdf(yeni_pdf)

    print("Maddelere bölünüyor ve ön işleniyor...")
    eski_maddeler = parse_maddeler(eski_text)
    yeni_maddeler = parse_maddeler(yeni_text)
    print(f" {len(eski_maddeler)} eski, {len(yeni_maddeler)} yeni madde bulundu.")

    print(" Benzerlik analizi yapılıyor...")
    df = tfidf_karsilastir(eski_maddeler, yeni_maddeler)
    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    print(" Sonuçlar kaydedildi: results.csv")

    print(" Zipf grafiği oluşturuluyor...")
    zipf_grafigi_olustur(eski_text + " " + yeni_text)

if __name__ == "__main__":
    main()
