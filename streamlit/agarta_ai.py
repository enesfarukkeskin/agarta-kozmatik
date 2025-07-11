import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import json
import requests
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Kozmetik Satış Tahmin Sistemi",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile stil düzenlemeleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B9D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .category-header {
        color: #4A90E2;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .ai-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        color: white;
    }
    .ai-response {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .upload-box {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .upload-box h3 {
        color: #1a252f;
        font-weight: bold;
    }
    .upload-box p {
        color: #2c3e50;
        font-weight: 500;
    }
    .upload-box li {
        color: #2c3e50;
        font-weight: 500;
    }
    .upload-box strong {
        color: #1a252f;
        font-weight: bold;
    }
    .upload-box em {
        color: #34495e;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# AI API fonksiyonları
def call_openai_api(api_key, prompt):
    """OpenAI API'yi çağırır"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system", 
                    "content": "Sen bir satış analisti ve iş danışmanısın. Kozmetik satış verilerini analiz edip, pratik ve uygulanabilir öneriler sunuyorsun. Türkçe yanıt ver."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Hatası: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def call_gemini_api(api_key, prompt):
    """Gemini API'yi çağırır"""
    try:
        # Yeni Gemini model adını kullan
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Sen bir satış analisti ve iş danışmanısın. Kozmetik satış verilerini analiz edip, pratik ve uygulanabilir öneriler sunuyorsun. Türkçe yanıt ver.\n\n{prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1500
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"API Hatası: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def generate_ai_analysis_prompt(prediction_data, prediction_type, selected_category=None):
    """AI analizi için prompt oluşturur"""
    
    if prediction_type == "📊 Genel Satış Tahmini":
        prompt = f"""
        Kozmetik satış tahmin analizi:
        
        ## Genel Satış Tahminleri:
        
        **Toplam Tahmin Sonuçları:**
        - Toplam tahmini satış: {prediction_data.get('total_quantity', 0):,.0f} adet
        - Tahmini ciro: {prediction_data.get('total_revenue', 0):,.0f} ₺
        - Tahmin periyodu: {prediction_data.get('prediction_days', 30)} gün
        - Günlük ortalama satış: {prediction_data.get('daily_average', 0):.1f} adet
        - Günlük ortalama ciro: {prediction_data.get('daily_revenue', 0):,.0f} ₺
        
        **Kategori Bazlı Dağılım:**
        """
        
        # Kategori detaylarını ekle
        if 'category_details' in prediction_data:
            for category, details in prediction_data['category_details'].items():
                prompt += f"""
        • {category}:
          - Günlük ortalama: {details.get('daily_avg', 0):.1f} adet
          - Toplam miktar: {details.get('total_quantity', 0):,.0f} adet
          - Tahmini ciro: {details.get('total_revenue', 0):,.0f} ₺
          - Toplam ciro içindeki payı: {details.get('revenue_share', 0):.1f}%
                """
        
        prompt += """
        
        Bu tahmin sonuçlarını analiz ederek:
        1. **Güçlü ve zayıf kategorileri** belirle
        2. **Satış artırma stratejileri** öner
        3. **Ürün karmasında yapılabilecek iyileştirmeler** öner
        4. **Pazarlama ve promosyon önerileri** sun
        5. **Risk faktörleri ve fırsatları** değerlendir
        
        Önerilerini maddeler halinde, pratik ve uygulanabilir şekilde sun.
        """
    
    else:  # Kategori bazlı tahmin
        prompt = f"""
        {selected_category} kategorisi için detaylı satış tahmin analizi:
        
        **Kategori Tahmin Sonuçları:**
        - Kategori: {selected_category}
        - Toplam tahmini satış: {prediction_data.get('total_quantity', 0):,.0f} adet
        - Günlük ortalama satış: {prediction_data.get('daily_avg', 0):.1f} adet
        - Tahmini ciro: {prediction_data.get('total_revenue', 0):,.0f} ₺
        - Tahmin periyodu: {prediction_data.get('prediction_days', 30)} gün
        - Minimum günlük satış: {prediction_data.get('min_daily', 0):.1f} adet
        - Maksimum günlük satış: {prediction_data.get('max_daily', 0):.1f} adet
        - Standart sapma: {prediction_data.get('std_dev', 0):.1f}
        
        Bu kategori için:
        1. **Satış performansını** değerlendir
        2. **Kategori bazında iyileştirme önerileri** sun
        3. **Bu kategoride satış artırma stratejileri** öner
        4. **Stok yönetimi önerileri** ver
        5. **Müşteri segmentasyonu ve hedefleme önerileri** sun
        6. **Fiyatlandırma stratejisi önerileri** ver
        
        Önerilerini bu kategori özelinde, maddeler halinde ve uygulanabilir şekilde sun.
        """
    
    return prompt

# Ana başlık
st.markdown('<h1 class="main-header">🧴 Kozmetik Satış Tahmin Sistemi</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Kontrol Paneli")
st.sidebar.markdown("---")

# AI Analiz Ayarları
st.sidebar.markdown("### 🤖 AI Analiz Ayarları")
enable_ai = st.sidebar.checkbox("🧠 AI Analiz Etkinleştir", help="Tahmin sonuçlarınızı AI ile analiz edin ve öneriler alın")

ai_provider = None
api_key = None

if enable_ai:
    ai_provider = st.sidebar.selectbox(
        "AI Sağlayıcı Seçin:",
        ["OpenAI (ChatGPT)", "Google Gemini"],
        help="Kullanmak istediğiniz AI servisini seçin"
    )
    
    if ai_provider == "OpenAI (ChatGPT)":
        api_key = st.sidebar.text_input(
            "🔑 OpenAI API Key:",
            type="password",
            help="OpenAI API anahtarınızı girin"
        )
        if api_key:
            st.sidebar.success("✅ OpenAI API Key girildi")
    
    else:  # Google Gemini
        api_key = st.sidebar.text_input(
            "🔑 Google Gemini API Key:",
            type="password",
            help="Google Gemini API anahtarınızı girin"
        )
        if api_key:
            st.sidebar.success("✅ Gemini API Key girildi")
    
    if not api_key:
        st.sidebar.warning("⚠️ AI analiz için API key gerekli")

st.sidebar.markdown("---")

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    try:
        with open('cosmetics_prophet_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Kategori modellerini bul
        models = {}
        
        # Eğer 'category_models' anahtarı varsa
        if 'category_models' in model_data:
            category_models = model_data['category_models']
            if isinstance(category_models, dict):
                models = category_models
        
        # Eğer direkt kategori isimleri anahtar olarak varsa
        elif any(cat in model_data for cat in CATEGORIES.keys()):
            for cat in CATEGORIES.keys():
                if cat in model_data:
                    models[cat] = model_data[cat]
        
        return models if models else None
        
    except FileNotFoundError:
        st.error("❌ Model dosyası bulunamadı! Lütfen 'cosmetics_prophet_models.pkl' dosyasının doğru konumda olduğundan emin olun.")
        return None
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata oluştu: {str(e)}")
        return None

# Veri yükleme ve işleme fonksiyonu
@st.cache_data
def process_uploaded_data(uploaded_file):
    try:
        # CSV dosyasını oku
        df = pd.read_csv(uploaded_file)
        
        # Sütun isimlerini normalize et (hem İngilizce hem Türkçe destekler)
        column_mapping = {}
        df_columns_lower = [col.lower().strip() for col in df.columns]
        
        # Tarih sütunu mapping
        for i, col in enumerate(df_columns_lower):
            if col in ['date', 'tarih', 'ds']:
                column_mapping[df.columns[i]] = 'date'
                break
        
        # Kategori sütunu mapping  
        for i, col in enumerate(df_columns_lower):
            if col in ['category', 'kategori', 'cat']:
                column_mapping[df.columns[i]] = 'category'
                break
        
        # Miktar sütunu mapping
        for i, col in enumerate(df_columns_lower):
            if col in ['quantity', 'adet', 'miktar', 'qty', 'y']:
                column_mapping[df.columns[i]] = 'quantity'
                break
        
        # Sütunları yeniden adlandır
        df = df.rename(columns=column_mapping)
        
        # Gerekli sütunları kontrol et
        required_columns = ['date', 'category', 'quantity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Gerekli sütunlar bulunamadı!")
            
            # Hangi sütunların mevcut olduğunu göster
            available_cols = list(df.columns)
            st.info(f"**Dosyanızdaki sütunlar:** {', '.join(available_cols)}")
            st.info("**Desteklenen sütun isimleri:**")
            st.write("• **Tarih:** date, tarih, ds")
            st.write("• **Kategori:** category, kategori, cat") 
            st.write("• **Miktar:** quantity, adet, miktar, qty, y")
            return None
        
        # Tarihi datetime'a çevir
        df['date'] = pd.to_datetime(df['date'])
        
        # Kategorileri temizle
        df['category'] = df['category'].str.strip()
        
        # Miktarı sayısal değere çevir
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        
        # NaN değerleri temizle
        df = df.dropna(subset=['date', 'category', 'quantity'])
        
        return df
        
    except Exception as e:
        st.error(f"❌ Veri işlenirken hata oluştu: {str(e)}")
        return None

# Kategori ismi normalize etme fonksiyonu
def normalize_category_name(category_name):
    """Kategori ismini snake_case formatına çevirir"""
    # Türkçe karakterleri değiştir
    replacements = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
    }
    
    normalized = category_name.lower().strip()
    for tr_char, en_char in replacements.items():
        normalized = normalized.replace(tr_char, en_char)
    
    # Boşlukları alt çizgi ile değiştir
    normalized = normalized.replace(' ', '_')
    
    return normalized

def reverse_normalize_category_name(snake_case_name):
    """snake_case formatından okunabilir formata çevirir"""
    # Alt çizgileri boşluk ile değiştir
    readable = snake_case_name.replace('_', ' ')
    
    # İlk harfleri büyük yap
    readable = ' '.join(word.capitalize() for word in readable.split())
    
    # Türkçe karakterleri geri çevir
    replacements = {
        'Cilt Bakim': 'Cilt Bakım',
        'Sac Bakim': 'Saç Bakım', 
        'Agiz Bakim': 'Ağız Bakım',
        'Vegan Urunler': 'Vegan Ürünler',
        'Avantajli Paketler': 'Avantajlı Paketler'
    }
    
    return replacements.get(readable, readable)

# Kategori bilgileri
CATEGORIES = {
    'Cilt Bakım': {'avg_daily': 63.4, 'avg_price': 122.5, 'icon': '🧴'},
    'Saç Bakım': {'avg_daily': 45.6, 'avg_price': 98.8, 'icon': '🧴'},
    'Ağız Bakım': {'avg_daily': 27.1, 'avg_price': 50.8, 'icon': '🦷'},
    'Vegan Ürünler': {'avg_daily': 18.0, 'avg_price': 139.3, 'icon': '🌱'},
    'Avantajlı Paketler': {'avg_daily': 14.5, 'avg_price': 207.1, 'icon': '📦'},
    'Sabun': {'avg_daily': 13.1, 'avg_price': 39.9, 'icon': '🧼'}
}

# Model yükleme
models = load_model()

if models is not None:
    st.sidebar.success("✅ Model başarıyla yüklendi!")
    
    # Veri yükleme alanı
    st.markdown("## 📁 Veri Yükleme")
    
    st.markdown("""
    <div class="upload-box">
        <h3>📊 Geçmiş Satış Verilerinizi Yükleyin</h3>
        <p>CSV dosyanız aşağıdaki sütunlardan birini içermelidir:</p>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>Tarih</strong>: date, tarih, ds (YYYY-MM-DD formatında)</li>
            <li><strong>Kategori</strong>: category, kategori, cat</li>
            <li><strong>Miktar</strong>: quantity, adet, miktar, qty, y</li>
        </ul>
        <p><em>İngilizce ve Türkçe sütun isimleri desteklenmektedir!</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "CSV dosyasını seçin",
        type=['csv'],
        help="Hem İngilizce hem Türkçe sütun isimleri desteklenir: tarih/date, kategori/category, adet/quantity"
    )
    
    if uploaded_file is not None:
        # Veriyi işle
        df = process_uploaded_data(uploaded_file)
        
        if df is not None:
            st.success("✅ Veri başarıyla yüklendi!")
            
            # Veri önizlemesi
            with st.expander("📋 Yüklenen Veri Önizlemesi"):
                st.write("**İlk 10 satır:**")
                st.dataframe(df.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📅 Toplam Gün", len(df['date'].unique()))
                with col2:
                    st.metric("🎨 Kategori Sayısı", len(df['category'].unique()))
                with col3:
                    st.metric("📦 Toplam Satış", f"{df['quantity'].sum():,}")
                
                st.write("**Mevcut Kategoriler:**")
                categories_in_data = df['category'].unique()
                st.write(", ".join(categories_in_data))
            
            # Tahmin seçenekleri
            st.sidebar.markdown("### 🎯 Tahmin Ayarları")
            
            # Tahmin türü seçimi
            prediction_type = st.sidebar.selectbox(
                "Tahmin Türü Seçin:",
                ["📊 Genel Satış Tahmini", "🎨 Kategori Bazlı Tahmin"]
            )
            
            # Tarih aralığı seçimi
            prediction_days = st.sidebar.slider("Kaç gün için tahmin yapılsın?", 7, 90, 30)
            
            # Başlangıç tarihi (son veri tarihinden sonra)
            last_date = df['date'].max()
            start_date = st.sidebar.date_input(
                "Tahmin başlangıç tarihi:",
                value=(last_date + timedelta(days=1)).date(),
                min_value=(last_date + timedelta(days=1)).date()
            )
            
            if prediction_type == "📊 Genel Satış Tahmini":
                st.markdown("## 📈 Genel Satış Tahmin Sonuçları")
                
                # Tahmin hesaplama
                total_predictions = {}
                total_revenue = 0
                total_quantity = 0
                
                # Kullanıcı verisindeki kategorileri kontrol et
                available_categories = df['category'].unique()
                
                # Kategori isimlerini normalize et ve eşleştir
                category_mapping = {}
                for user_category in available_categories:
                    normalized_user_cat = normalize_category_name(user_category)
                    if normalized_user_cat in models:
                        category_mapping[user_category] = normalized_user_cat
                
                st.write("**Kategori Eşleştirmeleri:**")
                for user_cat, model_cat in category_mapping.items():
                    st.write(f"• {user_cat} → {model_cat}")
                
                if not category_mapping:
                    st.warning("⚠️ Hiçbir kategori eşleştirilemedi!")
                    st.write("**Verinizdeki kategoriler:**")
                    for cat in available_categories:
                        normalized = normalize_category_name(cat)
                        st.write(f"• {cat} → {normalized}")
                    st.write("**Model kategorileri:**")
                    for cat in models.keys():
                        readable = reverse_normalize_category_name(cat)
                        st.write(f"• {cat} → {readable}")
                    st.stop()
                
                # Her kategori için tahmin yap
                for user_category, model_category in category_mapping.items():
                    # Gelecek tarihler oluştur
                    future_dates = pd.date_range(
                        start=start_date,
                        periods=prediction_days,
                        freq='D'
                    )
                    
                    future_df = pd.DataFrame({
                        'ds': future_dates
                    })
                    
                    # Tahmin yap
                    forecast = models[model_category].predict(future_df)
                    daily_avg = forecast['yhat'].mean()
                    daily_avg = max(0, daily_avg)  # Negatif değerleri sıfırla
                    
                    category_total = daily_avg * prediction_days
                    
                    # Kategori fiyat bilgisi (okunabilir ismi kullan)
                    readable_category = reverse_normalize_category_name(model_category)
                    if readable_category in CATEGORIES:
                        avg_price = CATEGORIES[readable_category]['avg_price']
                    else:
                        avg_price = 100  # Varsayılan fiyat
                            
                    category_revenue = category_total * avg_price
                    
                    total_predictions[user_category] = {
                        'daily_avg': daily_avg,
                        'total_quantity': category_total,
                        'total_revenue': category_revenue,
                        'forecast_data': forecast
                    }
                    
                    total_quantity += category_total
                    total_revenue += category_revenue
                
                if total_predictions:
                    # Özet metrikleri
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="🎯 Toplam Tahmin",
                            value=f"{int(total_quantity):,} adet",
                            delta=f"Günlük: {int(total_quantity/prediction_days)} adet"
                        )
                    
                    with col2:
                        st.metric(
                            label="💰 Tahmini Ciro",
                            value=f"{total_revenue:,.0f} ₺",
                            delta=f"Günlük: {total_revenue/prediction_days:,.0f} ₺"
                        )
                    
                    with col3:
                        st.metric(
                            label="📅 Tahmin Periyodu",
                            value=f"{prediction_days} gün",
                            delta=f"{start_date} - {start_date + timedelta(days=prediction_days-1)}"
                        )
                    
                    with col4:
                        avg_daily_revenue = total_revenue / prediction_days
                        st.metric(
                            label="📊 Ortalama Günlük",
                            value=f"{avg_daily_revenue:,.0f} ₺",
                            delta=f"{total_quantity/prediction_days:.1f} adet/gün"
                        )
                    
                    # AI Analiz Bölümü - Genel Tahmin
                    if enable_ai and api_key:
                        st.markdown("---")
                        st.markdown("## 🤖 AI Analiz ve Öneriler")
                        
                        # AI analizi için veri hazırlama
                        ai_data = {
                            'total_quantity': total_quantity,
                            'total_revenue': total_revenue,
                            'prediction_days': prediction_days,
                            'daily_average': total_quantity / prediction_days,
                            'daily_revenue': total_revenue / prediction_days,
                            'category_details': {}
                        }
                        
                        # Kategori detaylarını ekle
                        for category, data in total_predictions.items():
                            ai_data['category_details'][category] = {
                                'daily_avg': data['daily_avg'],
                                'total_quantity': data['total_quantity'],
                                'total_revenue': data['total_revenue'],
                                'revenue_share': (data['total_revenue'] / total_revenue) * 100
                            }
                        
                        # AI analizi yap
                        with st.spinner("🧠 AI analiz yapılıyor..."):
                            prompt = generate_ai_analysis_prompt(ai_data, prediction_type)
                            
                            if ai_provider == "OpenAI (ChatGPT)":
                                ai_response = call_openai_api(api_key, prompt)
                            else:
                                ai_response = call_gemini_api(api_key, prompt)
                        
                        # AI yanıtını göster
                        st.markdown("""
                        <div class="ai-box">
                            <h3>🤖 AI Analiz Sonuçları</h3>
                            <p>Tahmin sonuçlarınız AI tarafından analiz edildi. İşte öneriler:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI yanıtını düz metin olarak göster
                        st.write(ai_response)
                    
                    # Kategori bazlı dağılım
                    st.markdown("### 🎨 Kategori Bazlı Tahmin Dağılımı")
                    
                    # Pasta grafik için veri hazırlama
                    category_data = []
                    for category, data in total_predictions.items():
                        icon = CATEGORIES.get(category, {}).get('icon', '📦')
                        category_data.append({
                            'Kategori': f"{icon} {category}",
                            'Miktar': data['total_quantity'],
                            'Ciro': data['total_revenue']
                        })
                    
                    df_category = pd.DataFrame(category_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Miktar dağılımı
                        fig_quantity = px.pie(
                            df_category, 
                            values='Miktar', 
                            names='Kategori',
                            title="📦 Miktar Dağılımı (Adet)",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_quantity.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_quantity, use_container_width=True)
                    
                    with col2:
                        # Ciro dağılımı
                        fig_revenue = px.pie(
                            df_category, 
                            values='Ciro', 
                            names='Kategori',
                            title="💰 Ciro Dağılımı (₺)",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_revenue.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_revenue, use_container_width=True)
                    
                    # Detaylı tablo
                    st.markdown("### 📋 Detaylı Kategori Analizi")
                    
                    detailed_data = []
                    for category, data in total_predictions.items():
                        icon = CATEGORIES.get(category, {}).get('icon', '📦')
                        avg_price = CATEGORIES.get(category, {}).get('avg_price', 100)
                        
                        detailed_data.append({
                            'Kategori': f"{icon} {category}",
                            'Günlük Ortalama': f"{data['daily_avg']:.1f} adet",
                            'Toplam Miktar': f"{int(data['total_quantity']):,} adet",
                            'Ortalama Fiyat': f"{avg_price:.1f} ₺",
                            'Tahmini Ciro': f"{data['total_revenue']:,.0f} ₺",
                            'Ciro Payı': f"{(data['total_revenue']/total_revenue)*100:.1f}%"
                        })
                    
                    df_detailed = pd.DataFrame(detailed_data)
                    st.dataframe(df_detailed, use_container_width=True)
                    
                    # Zaman serisi grafiği
                    st.markdown("### 📈 Günlük Tahmin Trendi")
                    
                    # Tüm kategoriler için günlük tahminleri birleştir
                    fig_timeline = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    
                    for i, (category, data) in enumerate(total_predictions.items()):
                        forecast_df = data['forecast_data']
                        icon = CATEGORIES.get(category, {}).get('icon', '📦')
                        fig_timeline.add_trace(go.Scatter(
                            x=forecast_df['ds'],
                            y=forecast_df['yhat'],
                            name=f"{icon} {category}",
                            line=dict(color=colors[i % len(colors)], width=2),
                            mode='lines+markers'
                        ))
                    
                    fig_timeline.update_layout(
                        title="📊 Kategori Bazlı Günlük Satış Tahminleri",
                        xaxis_title="Tarih",
                        yaxis_title="Günlük Satış (Adet)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                else:
                    st.warning("⚠️ Verinizdeki kategoriler için eğitilmiş model bulunamadı!")
                    st.info(f"**Verinizdeki kategoriler:** {', '.join(available_categories)}")
                    st.info(f"**Mevcut model kategorileri:** {', '.join(models.keys())}")
            
            else:  # Kategori bazlı tahmin
                st.markdown("## 🎨 Kategori Bazlı Detaylı Tahmin")
                
                # Kullanıcı verisindeki kategorileri filtrele
                available_categories = df['category'].unique()
                
                # Kategori eşleştirmesi yap
                valid_categories = []
                category_mapping = {}
                
                for user_category in available_categories:
                    normalized_user_cat = normalize_category_name(user_category)
                    if normalized_user_cat in models:
                        valid_categories.append(user_category)
                        category_mapping[user_category] = normalized_user_cat
                
                if valid_categories:
                    # Kategori seçimi
                    selected_category = st.selectbox(
                        "Kategori seçin:",
                        valid_categories,
                        format_func=lambda x: f"{CATEGORIES.get(reverse_normalize_category_name(normalize_category_name(x)), {}).get('icon', '📦')} {x}"
                    )
                    
                    # Normalize edilmiş kategori ismini al
                    model_category = category_mapping[selected_category]
                    
                    # Gelecek tarihler oluştur
                    future_dates = pd.date_range(
                        start=start_date,
                        periods=prediction_days,
                        freq='D'
                    )
                    
                    future_df = pd.DataFrame({
                        'ds': future_dates
                    })
                    
                    # Tahmin yap
                    forecast = models[model_category].predict(future_df)
                    
                    # Negatif değerleri sıfırla
                    forecast['yhat'] = forecast['yhat'].clip(lower=0)
                    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                    
                    # Özet istatistikler
                    daily_avg = forecast['yhat'].mean()
                    total_quantity = forecast['yhat'].sum()
                    
                    # Kategori bilgilerini al
                    readable_category = reverse_normalize_category_name(model_category)
                    avg_price = CATEGORIES.get(readable_category, {}).get('avg_price', 100)
                    total_revenue = total_quantity * avg_price
                    
                    # Geçmiş veriden ortalama hesapla
                    historical_data = df[df['category'] == selected_category]
                    historical_avg = historical_data['quantity'].mean() if len(historical_data) > 0 else 0
                    
                    # Metrikleri göster
                    icon = CATEGORIES.get(readable_category, {}).get('icon', '📦')
                    st.markdown(f"### {icon} {selected_category} Tahmin Sonuçları")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="📦 Toplam Miktar",
                            value=f"{int(total_quantity):,} adet",
                            delta=f"Günlük: {daily_avg:.1f} adet"
                        )
                    
                    with col2:
                        st.metric(
                            label="💰 Tahmini Ciro",
                            value=f"{total_revenue:,.0f} ₺",
                            delta=f"Günlük: {total_revenue/prediction_days:,.0f} ₺"
                        )
                    
                    with col3:
                        st.metric(
                            label="💵 Ortalama Fiyat",
                            value=f"{avg_price:.1f} ₺",
                            delta="Kategori ortalaması"
                        )
                    
                    with col4:
                        if historical_avg > 0:
                            change_pct = ((daily_avg - historical_avg) / historical_avg) * 100
                            st.metric(
                                label="📊 Değişim",
                                value=f"{change_pct:+.1f}%",
                                delta=f"Geçmiş ort: {historical_avg:.1f}"
                            )
                        else:
                            st.metric(
                                label="📊 Tahmin",
                                value=f"{daily_avg:.1f}",
                                delta="Günlük ortalama"
                            )
                    
                    # AI Analiz Bölümü - Kategori Bazlı
                    if enable_ai and api_key:
                        st.markdown("---")
                        st.markdown("## 🤖 AI Analiz ve Öneriler")
                        
                        # AI analizi için veri hazırlama
                        ai_data = {
                            'total_quantity': total_quantity,
                            'daily_avg': daily_avg,
                            'total_revenue': total_revenue,
                            'prediction_days': prediction_days,
                            'min_daily': forecast['yhat'].min(),
                            'max_daily': forecast['yhat'].max(),
                            'std_dev': forecast['yhat'].std()
                        }
                        
                        # AI analizi yap
                        with st.spinner("🧠 AI analiz yapılıyor..."):
                            prompt = generate_ai_analysis_prompt(ai_data, prediction_type, selected_category)
                            
                            if ai_provider == "OpenAI (ChatGPT)":
                                ai_response = call_openai_api(api_key, prompt)
                            else:
                                ai_response = call_gemini_api(api_key, prompt)
                        
                        # AI yanıtını göster
                        st.markdown("""
                        <div class="ai-box">
                            <h3>🤖 AI Analiz Sonuçları</h3>
                            <p>Seçili kategori için AI analizi tamamlandı. İşte öneriler:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI yanıtını düz metin olarak göster
                        st.write(ai_response)
                    
                    # Tahmin grafiği
                    fig = go.Figure()
                    
                    # Ana tahmin çizgisi
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        name='Tahmin',
                        line=dict(color='#FF6B9D', width=3),
                        mode='lines+markers'
                    ))
                    
                    # Güven aralığı
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Güven Aralığı',
                        fillcolor='rgba(255, 107, 157, 0.2)'
                    ))
                    
                    fig.update_layout(
                        title=f"📈 {icon} {selected_category} - {prediction_days} Günlük Satış Tahmini",
                        xaxis_title="Tarih",
                        yaxis_title="Günlük Satış (Adet)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detaylı tablo
                    st.markdown("### 📋 Günlük Tahmin Detayları")
                    
                    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    forecast_display['ds'] = forecast_display['ds'].dt.strftime('%d/%m/%Y')
                    forecast_display['Tahmini Ciro'] = forecast_display['yhat'] * avg_price
                    
                    forecast_display = forecast_display.rename(columns={
                        'ds': 'Tarih',
                        'yhat': 'Tahmin (Adet)',
                        'yhat_lower': 'Alt Sınır',
                        'yhat_upper': 'Üst Sınır',
                        'Tahmini Ciro': 'Tahmini Ciro (₺)'
                    })
                    
                    # Sayısal değerleri yuvarla
                    for col in ['Tahmin (Adet)', 'Alt Sınır', 'Üst Sınır']:
                        forecast_display[col] = forecast_display[col].round(1)
                    forecast_display['Tahmini Ciro (₺)'] = forecast_display['Tahmini Ciro (₺)'].round(0)
                    
                    st.dataframe(forecast_display, use_container_width=True)
                    
                    # İstatistiksel özet
                    st.markdown("### 📊 İstatistiksel Özet")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class="info-box">
                        <h4>📈 Tahmin İstatistikleri</h4>
                        """, unsafe_allow_html=True)
                        
                        st.write(f"• **Minimum günlük satış**: {forecast['yhat'].min():.1f} adet")
                        st.write(f"• **Maksimum günlük satış**: {forecast['yhat'].max():.1f} adet")
                        st.write(f"• **Standart sapma**: {forecast['yhat'].std():.1f}")
                        st.write(f"• **Medyan**: {forecast['yhat'].median():.1f} adet")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="success-box">
                        <h4>💰 Ciro Analizi</h4>
                        """, unsafe_allow_html=True)
                        
                        daily_revenues = forecast['yhat'] * avg_price
                        st.write(f"• **Minimum günlük ciro**: {daily_revenues.min():,.0f} ₺")
                        st.write(f"• **Maksimum günlük ciro**: {daily_revenues.max():,.0f} ₺")
                        st.write(f"• **Ortalama günlük ciro**: {daily_revenues.mean():,.0f} ₺")
                        st.write(f"• **Toplam tahmini ciro**: {daily_revenues.sum():,.0f} ₺")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Verinizdeki kategoriler için eğitilmiş model bulunamadı!")
                    st.info(f"**Verinizdeki kategoriler:** {', '.join(df['category'].unique())}")
                    st.info(f"**Mevcut model kategorileri:** {', '.join(models.keys())}")
    
    else:
        # Veri yüklenmemişse örnek format göster
        st.markdown("### 📋 Desteklenen Veri Formatları")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**İngilizce Format:**")
            sample_data_en = pd.DataFrame({
                'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
                'category': ['Cilt Bakım', 'Saç Bakım', 'Cilt Bakım', 'Saç Bakım'],
                'quantity': [45, 32, 52, 28]
            })
            st.dataframe(sample_data_en, use_container_width=True)
        
        with col2:
            st.markdown("**Türkçe Format:**")
            sample_data_tr = pd.DataFrame({
                'tarih': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
                'kategori': ['Cilt Bakım', 'Saç Bakım', 'Cilt Bakım', 'Saç Bakım'],
                'adet': [45, 32, 52, 28]
            })
            st.dataframe(sample_data_tr, use_container_width=True)
        
        st.info("💡 **İpucu:** Verilerinizi yukarıdaki formatlardan herhangi biri gibi hazırlayın ve CSV olarak kaydedin.")
        
        # AI özellik açıklaması (veri yüklenmemişken)
        if enable_ai:
            st.markdown("---")
            st.markdown("""
            <div class="ai-box">
                <h3>🤖 AI Analiz Özelliği</h3>
                <p>Veri yükledikten sonra tahmin sonuçlarınızı AI ile analiz edebileceksiniz!</p>
                <ul style="text-align: left; margin: 10px 0;">
                    <li>📊 Satış performansı değerlendirmesi</li>
                    <li>💡 Satış artırma stratejileri</li>
                    <li>🎯 Kategori bazında öneriler</li>
                    <li>📈 Trend analizi ve öngörüler</li>
                    <li>💰 Ciro optimizasyon önerileri</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("❌ Model yüklenemedi. Lütfen model dosyasını kontrol edin.")
    
    st.markdown("""
    ### 📝 Model Dosyası Hakkında Bilgi
    
    Bu uygulama `cosmetics_prophet_models.pkl` dosyasını arar. Bu dosya aşağıdaki yapıda olmalıdır:
    
    ```python
    # Model dosyası içeriği
    {
        'Cilt Bakım': prophet_model_1,
        'Saç Bakım': prophet_model_2,
        'Ağız Bakım': prophet_model_3,
        'Vegan Ürünler': prophet_model_4,
        'Avantajlı Paketler': prophet_model_5,
        'Sabun': prophet_model_6
    }
    ```
    
    Lütfen model dosyasının uygulamayla aynı dizinde olduğundan emin olun.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🧴 Kozmetik Satış Tahmin Sistemi | Prophet ML Model ile güçlendirilmiştir</p>
    <p>📊 Veriye dayalı karar verme için tasarlanmıştır</p>
    <p>🤖 AI destekli analiz ve öneriler</p>
</div>
""", unsafe_allow_html=True)