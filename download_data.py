import os
import zipfile
import requests

# 1. labeled_anomalies.csv dosyasını indir
csv_url = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
print("labeled_anomalies.csv indiriliyor...")
r = requests.get(csv_url)
with open("labeled_anomalies.csv", "wb") as f:
    f.write(r.content)

# 2. Asıl telemetri verilerini içeren zip dosyasını indir
data_url = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
print("Telemetri verileri (data.zip) indiriliyor (yaklaşık 50 MB)...")
r = requests.get(data_url)
with open("data.zip", "wb") as f:
    f.write(r.content)

# 3. Zip dosyasını klasöre çıkart
print("Zip açılıyor...")
with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall(".")

print("İşlem tamamlandı! Veriler hazır.")