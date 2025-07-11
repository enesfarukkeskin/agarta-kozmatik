import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import seaborn as sns
from prophet import Prophet
import pickle

# Rastgelelik için seed
np.random.seed(42)
random.seed(42)

print("=== AGARTA KOZMETIK TİPİ E-TİCARET VERİSİ ÜRETİCİ ===")

# Tarih aralığı (2 yıl)
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

print(f"Veri aralığı: {start_date.date()} - {end_date.date()}")
print(f"Toplam gün sayısı: {len(date_range)}")

# Kozmetik sektörüne özel kategoriler (Agarta'dan esinlenerek)
categories = {
    'Cilt Bakım': {
        'products': ['Nemlendirici Krem', 'Temizleyici', 'Serum', 'Göz Kremi', 'Yüz Maskesi', 'Tonik', 'Peeling'],
        'price_range': (45, 180),
        'base_demand': 120,
        'seasonality': 'high_winter'  # Kışın daha çok satılır
    },
    'Saç Bakım': {
        'products': ['Şampuan', 'Saç Kremi', 'Saç Maskesi', 'Saç Serumu', 'Saç Yağı', 'Kuru Şampuan'],
        'price_range': (35, 150),
        'base_demand': 100,
        'seasonality': 'stable'
    },
    'Ağız Bakım': {
        'products': ['Diş Macunu', 'Ağız Gargarası', 'Diş Fırçası', 'Ağız Spreyi'],
        'price_range': (15, 80),
        'base_demand': 80,
        'seasonality': 'stable'
    },
    'Vegan Ürünler': {
        'products': ['Vegan Krem', 'Doğal Sabun', 'Organik Serum', 'Bitkisel Yağ'],
        'price_range': (60, 200),
        'base_demand': 60,
        'seasonality': 'trending_up'  # Artan trend
    },
    'Avantajlı Paketler': {
        'products': ['Set 2li', 'Set 3lü', 'Aile Paketi', 'Hediye Seti'],
        'price_range': (80, 300),
        'base_demand': 40,
        'seasonality': 'high_holidays'  # Özel günlerde yüksek
    },
    'Sabun': {
        'products': ['Katı Sabun', 'Sıvı Sabun', 'Antibakteriyel Sabun', 'Doğal Sabun'],
        'price_range': (12, 60),
        'base_demand': 90,
        'seasonality': 'pandemic_boost'  # Pandemi etkisi
    }
}

# Satış kanalları
sales_channels = ['Website', 'Mobile App', 'Amazon', 'Trendyol', 'Hepsiburada', 'N11']
channel_weights = [0.3, 0.25, 0.15, 0.15, 0.1, 0.05]

# Müşteri segmentleri
customer_segments = {
    'Genç Kadın (18-25)': {'ratio': 0.35, 'avg_order': 85, 'frequency': 'high'},
    'Yetişkin Kadın (26-40)': {'ratio': 0.40, 'avg_order': 120, 'frequency': 'medium'},
    'Orta Yaş (41-55)': {'ratio': 0.20, 'avg_order': 150, 'frequency': 'low'},
    'Erkek': {'ratio': 0.05, 'avg_order': 70, 'frequency': 'very_low'}
}

# Şehirler (Türkiye'nin büyük şehirleri)
cities = {
    'İstanbul': 0.25, 'Ankara': 0.12, 'İzmir': 0.08, 'Bursa': 0.05, 
    'Antalya': 0.04, 'Adana': 0.04, 'Konya': 0.03, 'Gaziantep': 0.03,
    'Mersin': 0.03, 'Diyarbakır': 0.02, 'Diğer': 0.31
}

def get_seasonal_multiplier(date, category_info):
    """Mevsimsel çarpan hesapla"""
    seasonality_type = category_info['seasonality']
    month = date.month
    
    if seasonality_type == 'high_winter':
        # Kış aylarında yüksek (cilt bakım)
        multipliers = [1.3, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.8, 1.0, 1.1, 1.2, 1.4]
    elif seasonality_type == 'high_holidays':
        # Özel günlerde yüksek (hediye setleri)
        multipliers = [0.8, 1.2, 1.0, 1.0, 1.3, 1.1, 0.9, 0.8, 1.0, 1.1, 1.4, 1.6]
    elif seasonality_type == 'trending_up':
        # Sürekli artan trend (vegan)
        base_trend = 0.8 + (date.dayofyear / 365.25) * 0.4
        multipliers = [base_trend] * 12
    elif seasonality_type == 'pandemic_boost':
        # Pandemi etkisi (sabun)
        if date.year == 2022 and date.month <= 6:
            multipliers = [1.5] * 12
        else:
            multipliers = [1.1] * 12
    else:  # stable
        multipliers = [1.0] * 12
        
    return multipliers[month - 1]

def get_special_day_boost(date):
    """Özel günler için artış"""
    boost = 0
    
    # Sevgililer Günü
    if date.month == 2 and date.day == 14:
        boost += 2.5
    
    # Kadınlar Günü  
    if date.month == 3 and date.day == 8:
        boost += 2.0
        
    # Anneler Günü (Mayıs ikinci pazarı)
    if date.month == 5 and date.day >= 8 and date.day <= 14 and date.weekday() == 6:
        boost += 3.0
        
    # Black Friday / Kasım indirimleri
    if date.month == 11 and date.day >= 20:
        boost += 2.2
        
    # Yılbaşı alışverişi
    if date.month == 12 and date.day >= 15:
        boost += 2.8
        
    # Ramazan öncesi temizlik (Mart-Nisan)
    if date.month in [3, 4] and date.day <= 15:
        boost += 1.5
        
    return boost

def generate_cosmetics_sales_data():
    """Ana veri üretici fonksiyon"""
    
    all_sales = []
    order_id_counter = 100000
    
    for date in date_range:
        # Günlük toplam sipariş sayısını belirle
        base_orders = 85  # Günlük ortalama sipariş
        
        # Haftalık etkiler
        weekday_multipliers = [1.1, 1.2, 1.1, 1.0, 1.3, 1.5, 0.7]  # Haftasonu düşük
        weekday_mult = weekday_multipliers[date.weekday()]
        
        # Özel gün etkisi
        special_boost = get_special_day_boost(date)
        
        # Trend etkisi (yavaş büyüme)
        days_from_start = (date - start_date).days
        trend_mult = 1 + (days_from_start * 0.0003)  # Yıllık %10 büyüme
        
        # Pandemi etkisi
        pandemic_mult = 1.0
        if date < datetime(2022, 4, 1):
            pandemic_mult = 1.3  # Online alışveriş artışı
        elif date < datetime(2022, 8, 1):
            pandemic_mult = 1.1
            
        # Günlük sipariş sayısı
        daily_orders = int(base_orders * weekday_mult * trend_mult * pandemic_mult * (1 + special_boost) + np.random.normal(0, 8))
        daily_orders = max(5, daily_orders)  # Minimum 5 sipariş
        
        # Her sipariş için veri üret
        for order_num in range(daily_orders):
            order_id_counter += 1
            
            # Kategori seç (ağırlıklı)
            category_weights = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]  # Cilt bakım en popüler
            selected_category = np.random.choice(list(categories.keys()), p=category_weights)
            category_info = categories[selected_category]
            
            # Ürün seç
            product = random.choice(category_info['products'])
            
            # Fiyat belirle
            min_price, max_price = category_info['price_range']
            base_price = random.uniform(min_price, max_price)
            
            # Mevsimsel fiyat etkisi (talep arttıkça fiyat da artar)
            seasonal_mult = get_seasonal_multiplier(date, category_info)
            price = base_price * (0.9 + seasonal_mult * 0.2)  # Fiyat mevsime göre %20 değişebilir
            
            # Miktar (genelde 1, bazen 2-3)
            quantity_probs = [0.75, 0.20, 0.04, 0.01]  # 1, 2, 3, 4+ adet
            quantity = np.random.choice([1, 2, 3, 4], p=quantity_probs)
            
            # Müşteri segmenti
            segment_name = np.random.choice(list(customer_segments.keys()), 
                                          p=[seg['ratio'] for seg in customer_segments.values()])
            segment_info = customer_segments[segment_name]
            
            # Satış kanalı
            channel = np.random.choice(sales_channels, p=channel_weights)
            
            # Şehir
            city = np.random.choice(list(cities.keys()), p=list(cities.values()))
            
            # İndirim (bazen)
            discount_rate = 0
            if random.random() < 0.15:  # %15 ihtimalle indirim
                discount_rate = random.choice([10, 15, 20, 25, 30])
            
            # Final fiyat
            unit_price = price * (1 - discount_rate / 100)
            total_amount = unit_price * quantity
            
            # Kargo durumu (genelde başarılı)
            status_probs = [0.85, 0.08, 0.04, 0.02, 0.01]
            status = np.random.choice(['Delivered', 'Shipped', 'Cancelled', 'Returned', 'Pending'], p=status_probs)
            
            # Sadece başarılı satışları dahil et
            if status in ['Delivered', 'Shipped']:
                actual_quantity = quantity
                actual_amount = total_amount
            else:
                actual_quantity = 0
                actual_amount = 0
            
            # Sipariş verisi
            order_data = {
                'order_id': f'AGR-{order_id_counter}',
                'date': date,
                'category': selected_category,
                'product': product,
                'quantity': actual_quantity,
                'unit_price': round(unit_price, 2),
                'total_amount': round(actual_amount, 2),
                'discount_rate': discount_rate,
                'customer_segment': segment_name,
                'sales_channel': channel,
                'city': city,
                'status': status,
                'day_of_week': date.strftime('%A'),
                'month': date.month,
                'quarter': f'Q{((date.month-1)//3)+1}',
                'is_weekend': date.weekday() >= 5,
                'is_holiday_season': date.month in [11, 12, 1],
                'seasonal_multiplier': round(seasonal_mult, 2)
            }
            
            all_sales.append(order_data)
    
    return pd.DataFrame(all_sales)

# Veri üret
print("\n=== VERİ ÜRETİLİYOR ===")
print("Bu işlem 1-2 dakika sürebilir...")

df_cosmetics = generate_cosmetics_sales_data()

print(f"✅ Toplam {len(df_cosmetics)} sipariş verisi oluşturuldu")

# Temel istatistikler
print("\n=== VERİ İSTATİSTİKLERİ ===")
print(f"Tarih aralığı: {df_cosmetics['date'].min()} - {df_cosmetics['date'].max()}")
print(f"Toplam sipariş: {len(df_cosmetics):,}")
print(f"Başarılı sipariş: {len(df_cosmetics[df_cosmetics['quantity'] > 0]):,}")
print(f"İptal/İade oranı: {(len(df_cosmetics[df_cosmetics['quantity'] == 0]) / len(df_cosmetics) * 100):.1f}%")

successful_orders = df_cosmetics[df_cosmetics['quantity'] > 0]
print(f"\nBaşarılı Siparişler:")
print(f"  Toplam ciro: {successful_orders['total_amount'].sum():,.0f} TL")
print(f"  Ortalama sipariş tutarı: {successful_orders['total_amount'].mean():.0f} TL")
print(f"  Günlük ortalama sipariş: {len(successful_orders) / len(date_range):.0f}")
print(f"  Günlük ortalama ciro: {successful_orders['total_amount'].sum() / len(date_range):,.0f} TL")

print("\n=== KATEGORİ ANALİZİ ===")
category_stats = successful_orders.groupby('category').agg({
    'quantity': 'sum',
    'total_amount': 'sum',
    'order_id': 'count'
}).round(0)
category_stats.columns = ['Toplam_Adet', 'Toplam_Ciro', 'Sipariş_Sayısı']
category_stats['Ortalama_Sipariş'] = (category_stats['Toplam_Ciro'] / category_stats['Sipariş_Sayısı']).round(0)
print(category_stats.sort_values('Toplam_Ciro', ascending=False))

print("\n=== KANAL ANALİZİ ===")
channel_stats = successful_orders.groupby('sales_channel').agg({
    'total_amount': 'sum',
    'order_id': 'count'
}).round(0)
channel_stats.columns = ['Toplam_Ciro', 'Sipariş_Sayısı']
channel_stats['Pay_%'] = (channel_stats['Toplam_Ciro'] / channel_stats['Toplam_Ciro'].sum() * 100).round(1)
print(channel_stats.sort_values('Toplam_Ciro', ascending=False))

print("\n=== MÜŞTERİ SEGMENTİ ANALİZİ ===")
segment_stats = successful_orders.groupby('customer_segment').agg({
    'total_amount': ['sum', 'mean'],
    'order_id': 'count'
}).round(0)
segment_stats.columns = ['Toplam_Ciro', 'Ortalama_Sipariş', 'Sipariş_Sayısı']
print(segment_stats.sort_values('Toplam_Ciro', ascending=False))

# Prophet için veri hazırla
print("\n=== PROPHET İÇİN VERİ HAZIRLAMA ===")

# Günlük toplam satış (miktar)
daily_sales_qty = successful_orders.groupby('date').agg({
    'quantity': 'sum',
    'total_amount': 'sum',
    'order_id': 'count'
}).reset_index()

# Prophet formatı
prophet_qty = daily_sales_qty[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
prophet_amount = daily_sales_qty[['date', 'total_amount']].rename(columns={'date': 'ds', 'total_amount': 'y'})
prophet_orders = daily_sales_qty[['date', 'order_id']].rename(columns={'date': 'ds', 'order_id': 'y'})

print(f"Prophet veri boyutu: {len(prophet_qty)} gün")
print(f"Günlük ortalama: {prophet_qty['y'].mean():.0f} adet")

# Kategori bazlı Prophet verileri
category_prophet_data = {}
for category in categories.keys():
    cat_data = successful_orders[successful_orders['category'] == category]
    cat_daily = cat_data.groupby('date')['quantity'].sum().reset_index()
    cat_daily = cat_daily.rename(columns={'date': 'ds', 'quantity': 'y'})
    
    # Eksik günleri 0 ile doldur
    full_dates = pd.DataFrame({'ds': date_range})
    cat_daily = full_dates.merge(cat_daily, on='ds', how='left').fillna(0)
    
    category_prophet_data[category] = cat_daily
    print(f"{category}: {len(cat_daily)} gün, ortalama {cat_daily['y'].mean():.1f} adet/gün")

print("\n=== DOSYALARI KAYDET ===")

# Ana dosyalar
df_cosmetics.to_csv('cosmetics_sales_full.csv', index=False, encoding='utf-8')
prophet_qty.to_csv('prophet_cosmetics_quantity.csv', index=False)
prophet_amount.to_csv('prophet_cosmetics_amount.csv', index=False)
prophet_orders.to_csv('prophet_cosmetics_orders.csv', index=False)

# Kategori dosyaları
for category, data in category_prophet_data.items():
    safe_filename = category.lower().replace(' ', '_').replace('ş', 's').replace('ç', 'c').replace('ğ', 'g').replace('ü', 'u').replace('ö', 'o').replace('ı', 'i')
    data.to_csv(f'prophet_cosmetics_{safe_filename}.csv', index=False)

print("✅ Tüm dosyalar kaydedildi!")

print("\n=== VERİ GÖRSELLEŞTİRME ===")

# Büyük görselleştirme
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Agarta Kozmetik Tarzı E-ticaret Satış Analizi', fontsize=16)

# 1. Günlük satış trendi
axes[0, 0].plot(prophet_qty['ds'], prophet_qty['y'])
axes[0, 0].set_title('Günlük Satış Miktarı Trendi')
axes[0, 0].set_ylabel('Adet')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Aylık ciro
monthly_revenue = successful_orders.groupby(successful_orders['date'].dt.to_period('M'))['total_amount'].sum()
axes[0, 1].plot(monthly_revenue.index.astype(str), monthly_revenue.values)
axes[0, 1].set_title('Aylık Ciro Trendi')
axes[0, 1].set_ylabel('Ciro (TL)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Kategori dağılımı
category_revenue = successful_orders.groupby('category')['total_amount'].sum().sort_values(ascending=True)
axes[1, 0].barh(category_revenue.index, category_revenue.values)
axes[1, 0].set_title('Kategoriye Göre Toplam Ciro')
axes[1, 0].set_xlabel('Ciro (TL)')

# 4. Kanal dağılımı
channel_revenue = successful_orders.groupby('sales_channel')['total_amount'].sum().sort_values(ascending=False)
axes[1, 1].pie(channel_revenue.values, labels=channel_revenue.index, autopct='%1.1f%%')
axes[1, 1].set_title('Satış Kanalları Dağılımı')

# 5. Haftalık pattern
weekly_pattern = successful_orders.groupby('day_of_week')['quantity'].mean()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pattern = weekly_pattern.reindex(day_order)
axes[2, 0].bar(weekly_pattern.index, weekly_pattern.values)
axes[2, 0].set_title('Haftalık Satış Paterni')
axes[2, 0].set_ylabel('Ortalama Günlük Adet')
axes[2, 0].tick_params(axis='x', rotation=45)

# 6. Aylık satış kutuları
monthly_sales = successful_orders.groupby([successful_orders['date'].dt.year, successful_orders['date'].dt.month])['quantity'].sum().reset_index()
monthly_sales['month_name'] = pd.to_datetime(monthly_sales[['date', 'month']].rename(columns={'date': 'year'})).dt.strftime('%b')
sns.boxplot(data=monthly_sales, x='month_name', y='quantity', ax=axes[2, 1])
axes[2, 1].set_title('Aylık Satış Dağılımı')
axes[2, 1].set_ylabel('Aylık Toplam Adet')
axes[2, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\n=== KOZMETİK SEKTÖRÜ ÖZELLİKLERİ ===")
print("✅ Üretilen verinin sektörel özellikleri:")
print("   🧴 Kategoriler: Cilt/Saç/Ağız bakım, Vegan, Avantajlı paketler")
print("   👩 Müşteri segmentleri: Yaş ve cinsiyet bazlı")
print("   🛒 Satış kanalları: Website, mobil, marketplace'ler")
print("   🎯 Özel günler: Kadınlar günü, Anneler günü, Sevgililer günü")
print("   📊 Mevsimsellik: Cilt bakım kışın, hediye setleri bayramlarda")
print("   💰 Fiyat aralıkları: Gerçekçi kozmetik fiyatları")
print("   🌟 Trendler: Vegan ürünlerde artış, pandemi etkisi")

print("\n=== DOSYA LİSTESİ ===")
print("📁 cosmetics_sales_full.csv (tam detay verisi)")
print("📁 prophet_cosmetics_quantity.csv (günlük miktar)")
print("📁 prophet_cosmetics_amount.csv (günlük ciro)")
print("📁 prophet_cosmetics_orders.csv (günlük sipariş sayısı)")
print("📁 prophet_cosmetics_[kategori].csv (kategori bazlı)")

print(f"\n🎉 Agarta Kozmetik tarzı {len(df_cosmetics):,} satış verisi hazır!")
print("🔮 Bu veri ile gerçekçi Prophet modelleri eğitebilirsiniz!")