# 📌 Metin Benzerliği Analizi (TF-IDF & Word2Vec)

Bu proje, Doğal Dil İşleme (NLP) dersi kapsamında gerçekleştirilmiştir. Amaç, ön işlenmiş metin verileri üzerinde TF-IDF ve Word2Vec modelleri kullanarak giriş metni ile benzer metinleri bulmak ve farklı yapılandırmalarla eğitilen modellerin başarılarını karşılaştırmaktır.

## 🔍 Kullanılan Veri Setleri

- `lemmatized.csv`: Leşleştirilmiş metinler
- `stemmed.csv`: Köklerine indirgenmiş metinler
- `tfidf_lemmatized.csv`, `tfidf_stemmed.csv`: TF-IDF vektör çıktıları
- 16 Word2Vec modeli:  
  - CBOW / SkipGram  
  - Pencere: 2 / 4  
  - Vektör boyutu: 100 / 300

## 🧪 Yöntem

### TF-IDF
- `TfidfVectorizer` ile oluşturulan vektörler kullanıldı.
- Giriş metni ile tüm metinler arasında cosine similarity hesaplandı.

### Word2Vec
- Her cümle, o cümledeki kelimelerin vektörlerinin ortalamasıyla temsil edildi.
- 16 farklı model için ayrı ayrı benzerlik hesaplamaları yapıldı.

## 📈 Değerlendirme Sonuçları

### En Benzer 5 Metin
Her model için ilk 5 en benzer metin, skorlarıyla birlikte elde edildi.

### Anlamsal Puanlama (1–5)
| Model Adı                      | Ortalama Skor |
| ------------------------------ | ------------- |
| w2v_lemma_cbow_win4_dim300     | 4.8           |
| tfidf_lemma                    | 4.4           |
| w2v_stem_skip_win4_dim300      | 4.2           |
| tfidf_stem                     | 4.0           |

### Jaccard Benzerlik Matrisi
Modellerin sıralama benzerliği analiz edildi. Örneğin:

- `w2v_lemma_cbow_win4_dim300` ile `tfidf_lemma` → Jaccard: 0.60
- Benzer yapılandırmaya sahip modellerin sıralama tutarlılığı yüksek çıktı.

## 💬 Yorumlar

- Word2Vec, anlamsal bağlamı yakalamada TF-IDF'e göre daha güçlü.
- CBOW mimarisi, genelde SkipGram'e göre daha başarılı sonuçlar verdi.
- TF-IDF daha hızlı ancak daha az semantik derinliğe sahip.

## 🚀 Öneriler

- Daha büyük veri setleri ile model başarısı artabilir.
- Stopword temizliği, POS etiketleme gibi ileri ön işleme adımları eklenebilir.
- Kullanıcı etkileşimli değerlendirme sistemleri geliştirilebilir.

---

### 📂 Çalıştırma Talimatları

1. Gerekli verileri `data/` klasörüne, modelleri `models/` klasörüne koyun.
2. Ana Python dosyasını çalıştırın:  
```bash
python metin_benzerlik.py
```
3. Sonuçlar terminalde skorlar ve yorumlarla birlikte gösterilir.

---
 
🎓 Ders: Doğal Dil İşleme  
📅 Tarih: Haziran 2025
