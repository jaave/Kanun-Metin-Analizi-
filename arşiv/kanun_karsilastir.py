import pdfplumber
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def save_text(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def load_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def parse_maddeler(text):
    parts = re.split(r'\bMADDE\s*\d+\b', text, flags=re.IGNORECASE)
    headers = re.findall(r'\bMADDE\s*\d+\b', text, flags=re.IGNORECASE)

    if not headers or len(parts) <= 1:
        return []

    maddeler = []
    for i in range(len(headers)):
        madde_no = headers[i].strip().upper()
        madde_icerik = parts[i + 1].strip()
        maddeler.append((madde_no, madde_icerik))

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

    # Eski metinlerde olup da yeni metinde eşleşmeyenleri tespit 
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

def main():
    eski_pdf = "eski.pdf"
    yeni_pdf = "yeni.pdf"

    if not os.path.exists(eski_pdf) or not os.path.exists(yeni_pdf):
        print("[!] PDF dosyaları klasörde bulunamadı.")
        return

    print(" PDF'ten metin çıkarılıyor...")
    eski_text = extract_text_from_pdf(eski_pdf)
    yeni_text = extract_text_from_pdf(yeni_pdf)

    save_text(eski_text, "eski_kanun.txt")
    save_text(yeni_text, "yeni_kanun.txt")

    print(" Maddelere bölünüyor...")
    eski_maddeler = parse_maddeler(eski_text)
    yeni_maddeler = parse_maddeler(yeni_text)

    print(f" {len(eski_maddeler)} eski, {len(yeni_maddeler)} yeni madde bulundu.")

    print("[3] Benzerlik analizi yapılıyor...")
    df = tfidf_karsilastir(eski_maddeler, yeni_maddeler)

    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    print("Tamamlandı! Sonuçlar 'results.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
