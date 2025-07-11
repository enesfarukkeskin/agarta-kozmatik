import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

print("=== MODEL KAYDETME SORUNU DÜZELTİCİ ===")
print("Grafik gösterme sırasında kesildiği için model kaydedilmemiş.")
print("Şimdi modeli yeniden eğitip kaydedeceğiz...")

# 1. VERİYİ HIZLICA YÜKLE VE MODELİ EĞİT
print("\n1. HIZLI MODEL EĞİTİMİ")

# Veriyi yükle
df = pd.read_csv('prophet_cosmetics_quantity.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Train/test ayır
split_date = '2023-10-01'
train = df[df['ds'] < split_date].copy()
test = df[df['ds'] >= split_date].copy()

print(f"✅ Veri yüklendi: {len(df)} gün")
print(f"🔵 Train: {len(train)} gün")
print(f"🟡 Test: {len(test)} gün")

# Tatiller
cosmetics_holidays = pd.DataFrame({
    'holiday': [
        'womens_day', 'womens_day', 
        'mothers_day_2022', 'mothers_day_2023',
        'valentine', 'valentine',
        'black_friday', 'black_friday',
        'new_year_shopping', 'new_year_shopping'
    ],
    'ds': [
        '2022-03-08', '2023-03-08',
        '2022-05-08', '2023-05-14',
        '2022-02-14', '2023-02-14',
        '2022-11-25', '2023-11-24',
        '2022-12-25', '2023-12-25'
    ],
    'lower_window': [-2, -2, -2, -2, -1, -1, -3, -3, -5, -5],
    'upper_window': [2, 2, 2, 2, 1, 1, 1, 1, 5, 5]
})
cosmetics_holidays['ds'] = pd.to_datetime(cosmetics_holidays['ds'])

# 2. TEMEL MODEL EĞİT
print("\n2. TEMEL MODELİ EĞİT")
model_basic = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    holidays_prior_scale=10,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    interval_width=0.95,
    holidays=cosmetics_holidays
)

print("🔄 Temel model eğitiliyor...")
model_basic.fit(train)

# Test performansı
future_basic = model_basic.make_future_dataframe(periods=len(test))
forecast_basic = model_basic.predict(future_basic)
test_forecast_basic = forecast_basic.tail(len(test))

mae_basic = mean_absolute_error(test['y'], test_forecast_basic['yhat'])
rmse_basic = np.sqrt(mean_squared_error(test['y'], test_forecast_basic['yhat']))
mape_basic = np.mean(np.abs((test['y'] - test_forecast_basic['yhat']) / test['y'])) * 100

print(f"📈 Temel Model: MAE={mae_basic:.1f}, RMSE={rmse_basic:.1f}, MAPE={mape_basic:.1f}%")

# 3. GELİŞMİŞ MODEL EĞİT
print("\n3. GELİŞMİŞ MODELİ EĞİT")
model_advanced = Prophet(
    changepoint_prior_scale=0.08,
    seasonality_prior_scale=15,
    holidays_prior_scale=15,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    interval_width=0.95,
    holidays=cosmetics_holidays
)

model_advanced.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_advanced.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

print("🔄 Gelişmiş model eğitiliyor...")
model_advanced.fit(train)

forecast_advanced = model_advanced.predict(future_basic)
test_forecast_advanced = forecast_advanced.tail(len(test))

mae_advanced = mean_absolute_error(test['y'], test_forecast_advanced['yhat'])
rmse_advanced = np.sqrt(mean_squared_error(test['y'], test_forecast_advanced['yhat']))
mape_advanced = np.mean(np.abs((test['y'] - test_forecast_advanced['yhat']) / test['y'])) * 100

print(f"📈 Gelişmiş Model: MAE={mae_advanced:.1f}, RMSE={rmse_advanced:.1f}, MAPE={mape_advanced:.1f}%")

# En iyi modeli seç
if mae_advanced < mae_basic:
    best_model = model_advanced
    best_forecast = forecast_advanced
    best_name = "Gelişmiş"
    best_performance = {'MAE': mae_advanced, 'RMSE': rmse_advanced, 'MAPE': mape_advanced}
    print("🏆 Gelişmiş model daha iyi!")
else:
    best_model = model_basic
    best_forecast = forecast_basic
    best_name = "Temel"
    best_performance = {'MAE': mae_basic, 'RMSE': rmse_basic, 'MAPE': mape_basic}
    print("🏆 Temel model yeterli!")

# 4. KATEGORİ MODELLERİ EĞİT
print("\n4. KATEGORİ MODELLERİNİ EĞİT")
categories = ['cilt_bakim', 'sac_bakim', 'agiz_bakim', 'vegan_urunler', 'avantajli_paketler', 'sabun']
category_models = {}
category_performance = {}

for category in categories:
    try:
        print(f"📋 {category.replace('_', ' ').title()} eğitiliyor...")
        
        cat_df = pd.read_csv(f'prophet_cosmetics_{category}.csv')
        cat_df['ds'] = pd.to_datetime(cat_df['ds'])
        
        cat_train = cat_df[cat_df['ds'] < split_date]
        cat_test = cat_df[cat_df['ds'] >= split_date]
        
        cat_model = Prophet(
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=15,
            holidays_prior_scale=10,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            holidays=cosmetics_holidays
        )
        
        cat_model.fit(cat_train)
        
        cat_future = cat_model.make_future_dataframe(periods=len(cat_test))
        cat_forecast = cat_model.predict(cat_future)
        cat_test_forecast = cat_forecast.tail(len(cat_test))
        
        cat_mae = mean_absolute_error(cat_test['y'], cat_test_forecast['yhat'])
        cat_mape = np.mean(np.abs((cat_test['y'] - cat_test_forecast['yhat']) / np.maximum(cat_test['y'], 1))) * 100
        
        category_models[category] = cat_model
        category_performance[category] = {'MAE': cat_mae, 'MAPE': cat_mape}
        
        print(f"   ✅ MAE: {cat_mae:.1f}, MAPE: {cat_mape:.1f}%")
        
    except Exception as e:
        print(f"   ❌ {category} hata: {str(e)}")

# 5. GELECEK TAHMİNLERİ YAP
print("\n5. GELECEK TAHMİNLERİ OLUŞTUR")
future_30 = best_model.make_future_dataframe(periods=30)
forecast_30 = best_model.predict(future_30)
future_predictions = forecast_30.tail(30)

monthly_total = future_predictions['yhat'].sum()
print(f"🔮 30 günlük tahmin: {int(monthly_total):,} adet")

# Kategori tahminleri
category_predictions = {}
for category, model in category_models.items():
    try:
        cat_future_30 = model.make_future_dataframe(periods=30)
        cat_forecast_30 = model.predict(cat_future_30)
        cat_pred_total = cat_forecast_30.tail(30)['yhat'].sum()
        category_predictions[category] = max(0, int(cat_pred_total))
    except:
        category_predictions[category] = 0

print(f"📋 Kategori toplamı: {sum(category_predictions.values()):,} adet")

# 6. MODELİ KAYDET - MANUEL OLARAK
print("\n6. MODELİ KAYDET")

models_dict = {
    'best_model': best_model,
    'best_model_name': best_name,
    'basic_model': model_basic,
    'advanced_model': model_advanced,
    'category_models': category_models,
    'performance': {
        'basic': {'MAE': mae_basic, 'RMSE': rmse_basic, 'MAPE': mape_basic},
        'advanced': {'MAE': mae_advanced, 'RMSE': rmse_advanced, 'MAPE': mape_advanced},
        'best': best_performance,
        'categories': category_performance
    },
    'predictions': {
        'next_30_days': future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
        'category_totals': category_predictions,
        'total_forecast': int(monthly_total),
        'revenue_forecast': int(monthly_total * 142)
    },
    'training_info': {
        'train_period': f"{train['ds'].min().date()} - {train['ds'].max().date()}",
        'test_period': f"{test['ds'].min().date()} - {test['ds'].max().date()}",
        'total_data_points': len(df),
        'model_created': pd.Timestamp.now(),
        'sector': 'Cosmetics E-commerce',
        'holidays_count': len(cosmetics_holidays)
    }
}

# PKL dosyasını kaydet
with open('cosmetics_prophet_models.pkl', 'wb') as f:
    pickle.dump(models_dict, f)

print("✅ cosmetics_prophet_models.pkl kaydedildi!")

# CSV tahminleri kaydet
future_predictions_clean = future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
future_predictions_clean['ds'] = future_predictions_clean['ds'].dt.date
future_predictions_clean.columns = ['Tarih', 'Tahmin', 'Alt_Sinir', 'Ust_Sinir']
future_predictions_clean['Tahmin'] = future_predictions_clean['Tahmin'].astype(int)
future_predictions_clean['Alt_Sinir'] = future_predictions_clean['Alt_Sinir'].astype(int)
future_predictions_clean['Ust_Sinir'] = future_predictions_clean['Ust_Sinir'].astype(int)
future_predictions_clean.to_csv('cosmetics_future_30days.csv', index=False)

print("✅ cosmetics_future_30days.csv kaydedildi!")

# 7. MODEL KULLANIM FONKSİYONU
def load_and_use_model():
    """Kaydedilen modeli yükle ve kullan"""
    try:
        with open('cosmetics_prophet_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        print(f"\n✅ Model yüklendi!")
        print(f"🏆 En iyi model: {models['best_model_name']}")
        print(f"📊 Doğruluk: {models['performance']['best']['MAPE']:.1f}% MAPE")
        print(f"📅 Oluşturulma: {models['training_info']['model_created'].strftime('%Y-%m-%d %H:%M')}")
        
        return models
    except:
        print("❌ Model dosyası bulunamadı!")
        return None

def make_quick_prediction(days=7):
    """Hızlı tahmin yap"""
    models = load_and_use_model()
    if not models:
        return
    
    best_model = models['best_model']
    future = best_model.make_future_dataframe(periods=days)
    forecast = best_model.predict(future)
    
    future_pred = forecast.tail(days)
    print(f"\n🔮 Gelecek {days} günün tahminleri:")
    for i, row in future_pred.iterrows():
        date = row['ds'].strftime('%Y-%m-%d (%A)')
        pred = int(row['yhat'])
        print(f"   {date}: {pred:,} adet")
    
    total = future_pred['yhat'].sum()
    print(f"\n📊 Toplam {days} gün: {int(total):,} adet")
    print(f"💰 Tahmini ciro: {int(total * 142):,} TL")

# 8. DOSYA KONTROLÜ
import os
print(f"\n8. DOSYA KONTROLÜ")
files_to_check = [
    'cosmetics_prophet_models.pkl',
    'cosmetics_future_30days.csv',
    'prophet_cosmetics_quantity.csv'
]

for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024  # KB
        print(f"✅ {file} ({size:.1f} KB)")
    else:
        print(f"❌ {file} bulunamadı!")

# 9. ÖZET RAPOR
print(f"\n" + "="*60)
print("📊 KOZMETİK PROPHET MODEL ÖZET RAPORU")
print("="*60)
print(f"🏆 En İyi Model: {best_name}")
print(f"📈 Model Doğruluğu: {best_performance['MAPE']:.1f}% MAPE")
print(f"📊 Kategori Sayısı: {len(category_models)}")
print(f"🔮 30 Gün Tahmini: {int(monthly_total):,} adet")
print(f"💰 Tahmini Ciro: {int(monthly_total * 142):,} TL")
print(f"📁 Model Dosyası: cosmetics_prophet_models.pkl")
print(f"📄 Tahmin Dosyası: cosmetics_future_30days.csv")

print(f"\n🎯 MODEL KULLANIM ÖRNEĞİ:")
print("```python")
print("# Modeli yükle")
print("import pickle")
print("with open('cosmetics_prophet_models.pkl', 'rb') as f:")
print("    models = pickle.load(f)")
print("")
print("# 7 günlük tahmin yap")
print("future = models['best_model'].make_future_dataframe(periods=7)")
print("forecast = models['best_model'].predict(future)")
print("print(forecast.tail(7)[['ds', 'yhat']])")
print("```")

print(f"\n✅ MODEL BAŞARIYLA KAYDEDİLDİ!")
print(f"💡 Artık 'tree' komutu ile cosmetics_prophet_models.pkl dosyasını görebilirsiniz!")

# Test et
print(f"\n🧪 HIZLI TEST:")
make_quick_prediction(3)