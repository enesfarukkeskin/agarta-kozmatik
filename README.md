# 🧴 Agarta Kozmetik - Satış Tahmin ve Analiz Sistemi

Bu proje, kozmetik e-ticaret sektörü için geliştirilmiş kapsamlı bir **satış tahmini ve analizi sistemi**dir. Facebook Prophet makine öğrenmesi algoritması kullanarak gelecekteki satış verilerini tahmin eder ve iş kararlarını destekleyici analizler sunar.

## 📊 Proje Hakkında

### 🎯 Amaç
- Kozmetik ürünlerin günlük satış tahminlerini yapmak
- Kategori bazlı satış analizi ve tahminleri sunmak
- Mevsimsel satış trendlerini analiz etmek
- İş geliştirme ve stok yönetimi için veri odaklı kararlar vermek
- AI destekli satış analizleri ve öneriler sunmak

### 🛠️ Teknolojiler
- **Python** - Ana programlama dili
- **Facebook Prophet** - Zaman serisi tahmini için makine öğrenmesi
- **Streamlit** - İnteraktif web arayüzü
- **Plotly** - İnteraktif grafikler ve görselleştirme
- **Pandas & NumPy** - Veri işleme ve analiz
- **Scikit-learn** - Model performans metrikleri
- **OpenAI/Gemini API** - AI destekli analiz ve öneriler

## 🏗️ Proje Yapısı

```
agarta-kozmatik/
├── 📊 Veri Dosyaları
│   ├── cosmetics_sales_full.csv          # Ana satış verisi
│   ├── prophet_cosmetics_quantity.csv     # Günlük toplam satış miktarı
│   ├── prophet_cosmetics_amount.csv       # Günlük toplam ciro
│   ├── prophet_cosmetics_orders.csv       # Günlük sipariş sayısı
│   └── prophet_cosmetics_[kategori].csv   # Kategori bazlı veriler
│
├── 🤖 Model ve Tahmin Dosyaları
│   ├── cosmetics_prophet_models.pkl       # Eğitilmiş Prophet modelleri
│   └── cosmetics_future_30days.csv        # 30 günlük tahminler
│
├── 🧠 Temel Python Dosyaları
│   ├── data.py                           # Veri üretimi ve preprocessing
│   └── training.py                       # Model eğitimi ve kaydetme
│
├── 🌐 Streamlit Web Uygulaması
│   ├── agarta.py                         # Temel tahmin arayüzü
│   ├── agarta_ai.py                      # AI destekli gelişmiş analiz
│   └── generate_data_for_test.py         # Test verisi üretimi
│
└── README.md                             # Bu dosya
```

## 🛍️ Ürün Kategorileri

Sistem aşağıdaki kozmetik kategorilerini destekler:

| Kategori | Ürün Örnekleri | Ortalama Fiyat |
|----------|----------------|----------------|
| **Cilt Bakım** | Nemlendirici, Serum, Temizleyici, Maske | ₺45 - ₺180 |
| **Saç Bakım** | Şampuan, Saç Kremi, Maske, Yağ | ₺35 - ₺150 |
| **Ağız Bakım** | Diş Macunu, Gargarası, Fırça | ₺15 - ₺80 |
| **Vegan Ürünler** | Doğal Sabun, Organik Serum, Bitkisel Yağ | ₺60 - ₺200 |
| **Avantajlı Paketler** | 2'li Set, 3'lü Set, Hediye Kutuları | ₺80 - ₺300 |
| **Sabun** | Katı/Sıvı Sabun, Antibakteriyel | ₺12 - ₺60 |

## 🔧 Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
pip install streamlit pandas numpy prophet plotly scikit-learn matplotlib seaborn requests
```

### 2. Projeyi İndirin
```bash
git clone [repo-url]
cd agarta-kozmatik
```

### 3. Veri Hazırlama
```bash
# Örnek veri üretmek için
python data.py
```

### 4. Model Eğitimi
```bash
# Prophet modellerini eğitmek için
python training.py
```

### 5. Web Uygulamasını Başlatma
```bash
# Temel tahmin arayüzü için
streamlit run streamlit/agarta.py

# AI destekli gelişmiş analiz için
streamlit run streamlit/agarta_ai.py
```

## 📈 Özellikler

### 🎯 Temel Tahmin Özellikleri
- **Günlük Satış Tahmini**: 1-365 gün arası tahmin yapabilir
- **Kategori Bazlı Tahmin**: Her kategori için ayrı tahmin modeli
- **Mevsimsel Analiz**: Tatil ve özel günlerin etkisini hesaplar
- **Güven Aralıkları**: Tahminler için alt ve üst sınır değerleri
- **Model Performansı**: MAE, RMSE, MAPE metrikleri ile doğruluk ölçümü

### 🤖 AI Destekli Analiz (agarta_ai.py)
- **OpenAI GPT-3.5/4 Entegrasyonu**: Akıllı satış analizi
- **Google Gemini AI**: Alternatif AI analiz motoru
- **Otomatik Rapor Üretimi**: AI destekli iş önerileri
- **Veri Yükleme**: CSV dosyası ile kendi verilerinizi analiz etme
- **İnteraktif Grafikler**: Plotly ile detaylı görselleştirme

### 📊 Analiz Türleri
1. **Genel Satış Tahmini**: Toplam satış ve ciro tahmini
2. **Kategori Analizi**: Kategori bazlı detaylı analiz
3. **Mevsimsel Trend**: Yıllık ve aylık satış paternleri
4. **Performans Analizi**: Model doğruluğu ve güvenilirlik
5. **İş Önerileri**: AI destekli stratejik öneriler

## 🎮 Kullanım Örnekleri

### Temel Tahmin Yapma
```python
# Model yükleme
import pickle
with open('cosmetics_prophet_models.pkl', 'rb') as f:
    models = pickle.load(f)

# 7 günlük tahmin
future = models['best_model'].make_future_dataframe(periods=7)
forecast = models['best_model'].predict(future)
```

### Kategori Bazlı Tahmin
```python
# Cilt bakım kategorisi için tahmin
cilt_model = models['category_models']['cilt_bakim'] 
future_cilt = cilt_model.make_future_dataframe(periods=30)
forecast_cilt = cilt_model.predict(future_cilt)
```

## 📊 Model Performansı

Sistem, farklı model türleri kullanarak en iyi sonucu seçer:

| Model Türü | Özellikler | Kullanım Alanı |
|------------|------------|-----------------|
| **Temel Model** | Standart Prophet | Genel tahminler |
| **Gelişmiş Model** | Özel sezonalite + tatiller | Detaylı analiz |
| **Kategori Modelleri** | Kategori özel ayarlar | Ürün bazlı tahmin |

### Performans Metrikleri
- **MAE (Mean Absolute Error)**: Ortalama mutlak hata
- **RMSE (Root Mean Square Error)**: Kök ortalama kare hata  
- **MAPE (Mean Absolute Percentage Error)**: Ortalama mutlak yüzde hata

## 🎯 İş Değeri ve Faydalar

### 📈 Satış ve Pazarlama
- **Stok Optimizasyonu**: Doğru miktarda stok tutma
- **Kampanya Planlama**: Yüksek satış dönemlerini öngörme
- **Bütçe Tahmini**: Gelecek dönem ciro projeksiyonları
- **Ürün Karması**: Hangi kategorilere odaklanacağını belirleme

### 💰 Finansal Fayda
- **Stok Maliyeti Azaltma**: Fazla stok tutmayı önleme
- **Satış Artışı**: Doğru zamanda doğru ürünü sunma
- **Nakit Akışı**: Gelecek gelirleri öngörme
- **ROI Optimizasyonu**: Yatırım getirilerini maksimize etme

### 🎯 Operasyonel Avantajlar
- **Tedarik Zinciri**: Tedarikçilerle daha iyi planlama
- **İnsan Kaynakları**: Yoğun dönemler için personel planlama
- **Lojistik**: Depo ve kargo kapasitesi planlama

## 🔮 AI Analiz Özellikleri

### 🤖 Desteklenen AI Modeller
- **OpenAI GPT-3.5-turbo**: Hızlı ve uygun maliyetli analiz
- **OpenAI GPT-4**: Daha detaylı ve karmaşık analizler
- **Google Gemini**: Alternatif AI motor

### 📋 AI Analiz Türleri
1. **Trend Analizi**: Satış trendlerinin yorumlanması
2. **Sezonalite Değerlendirmesi**: Mevsimsel paternlerin analizi
3. **Risk Değerlendirmesi**: Potansiyel riskler ve fırsatlar
4. **Strateji Önerileri**: Somut iş kararı önerileri
5. **Karşılaştırmalı Analiz**: Kategoriler arası performans karşılaştırması

## 🚀 Gelecek Geliştirmeler

### Planlanan Özellikler
- [ ] Real-time veri entegrasyonu
- [ ] Mobil uygulama desteği
- [ ] Otomatik raporlama sistemi
- [ ] E-ticaret platform entegrasyonları (Shopify, WooCommerce)
- [ ] Advanced dashboard'lar
- [ ] Çoklu dil desteği
- [ ] API servisleri

### Teknik Geliştirmeler
- [ ] Docker konteynerizasyonu
- [ ] Cloud deployment (AWS, Azure)
- [ ] Database entegrasyonu (PostgreSQL, MongoDB)
- [ ] Automated model retraining
- [ ] A/B testing framework

## 🤝 Katkıda Bulunma

Bu projeye katkıda bulunmak için:

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 İletişim ve Destek

- **Geliştirici**: [Geliştiricinin Adı]
- **E-posta**: [E-posta adresi]
- **LinkedIn**: [LinkedIn profili]
- **GitHub**: [GitHub profili]

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasını inceleyin.

## 🙏 Teşekkürler

- **Facebook Prophet** - Zaman serisi tahmin algoritması
- **Streamlit** - Kolay web uygulaması geliştirme
- **Plotly** - İnteraktif görselleştirme
- **OpenAI & Google** - AI analiz desteği

---

### 📚 Ek Kaynaklar

- [Facebook Prophet Dokümantasyonu](https://facebook.github.io/prophet/)
- [Streamlit Dokümantasyonu](https://docs.streamlit.io/)
- [Plotly Dokümantasyonu](https://plotly.com/python/)
- [Zaman Serisi Analizi Rehberi](https://otexts.com/fpp3/)

**Not**: Bu sistem demo amaçlı olarak geliştirilmiştir. Gerçek üretim ortamında kullanımdan önce verilerinizi ve iş gereksinimlerinizi değerlendirin.