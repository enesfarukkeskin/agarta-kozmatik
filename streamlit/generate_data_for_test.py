import csv
import random
from datetime import datetime, timedelta

# Kategoriler (numaralandırılmış)
category_map = {
    1: "Cilt Bakım",
    2: "Saç Bakım",
    3: "Ağız Bakım",
    4: "Vegan Ürünler",
    5: "Avantajlı Paketler",
    6: "Sabun"
}

# Kullanıcıya seçim menüsü göster
print("Lütfen bir kategori seçin:")
for key, value in category_map.items():
    print(f"{key}. {value}")

try:
    choice = int(input("Seçiminiz (1-6): "))
    selected_category = category_map[choice]
except (ValueError, KeyError):
    print("Geçersiz seçim. Lütfen 1 ile 6 arasında bir sayı girin.")
    exit()

# Rastgele tarih üretimi
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()

# CSV oluştur
filename = selected_category.replace(" ", "_") + ".csv"

with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["tarih", "kategori", "adet", "ciro"])

    for _ in range(300):  # 300 satır veri üret
        tarih = random_date(start_date, end_date).strftime("%Y-%m-%d")
        adet = random.randint(1, 20)
        birim_fiyat = random.uniform(10, 300)
        ciro = round(adet * birim_fiyat, 2)
        writer.writerow([tarih, selected_category, adet, ciro])

print(f"{filename} başarıyla oluşturuldu.")
