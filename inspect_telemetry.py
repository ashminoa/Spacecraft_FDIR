import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 1. Anomali etiket dosyasını oku
df = pd.read_csv('labeled_anomalies.csv')

# SMAP uydusuna ait ve Güç/Akım (P-) ile ilgili olan ilk kanalı seçelim
smap_p_channels = df[(df['spacecraft'] == 'SMAP') & (df['chan_id'].str.startswith('P-'))]
print("SMAP Güç/Akım Kanalları:")
print(smap_p_channels[['chan_id', 'anomaly_sequences', 'class']].head())

# İlk kanalın kimliğini ve anomali aralığını alalım (P-1)
target_channel = smap_p_channels.iloc[0]['chan_id']
anomaly_ranges = ast.literal_eval(smap_p_channels.iloc[0]['anomaly_sequences'])

# 2. Test telemetri verisini yükle
test_data = np.load(f'data/test/{target_channel}.npy')
telemetry_values = test_data[:, 0]  # İlk sütun: normalize edilmiş telemetri sinyali

# 3. Görselleştirme
plt.figure(figsize=(12, 4))
plt.plot(telemetry_values, label=f'{target_channel} Telemetri Sinyali', color='tab:blue', linewidth=1)

# Gerçek anomali bölgesini kırmızı ile işaretle
for start, end in anomaly_ranges:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Gerçek Anomali Bölgesi')

plt.title(f'NASA SMAP Görevi - {target_channel} Kanalı Telemetri Verisi ve Arıza Bölgesi')
plt.xlabel('Zaman Adımı (Dakika)')
plt.ylabel('Normalize Değer [-1, 1]')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()