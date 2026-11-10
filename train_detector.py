import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

# 1. Parameterkonfiguration
WINDOW_SIZE = 50   # Sequenzlänge (Vergangene 50 Minuten)
EPOCHS = 10        # Trainingsiterationen
BATCH_SIZE = 128
LR = 0.005

# 2. Laden der Telemetriedaten
train_raw = np.load('data/train/P-1.npy')[:, 0]  # Physikalische Telemetrieserie
test_raw = np.load('data/test/P-1.npy')[:, 0]

# 3. Vorbereitung der Vorhersagesequenzen: X (Schritte t-49 bis t) -> y (Schritt t+1)
def create_forecasting_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X)[..., np.newaxis], np.array(y)

X_train, y_train = create_forecasting_data(train_raw, WINDOW_SIZE)
X_test, y_test = create_forecasting_data(test_raw, WINDOW_SIZE)

# 4. Leichtes LSTM-Modell (NASA JPL / Hundman et al., 2018)
class TelemetryLSTM(nn.Module):
    def __init__(self):
        super(TelemetryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Zustand des letzten Zeitschritts
        return out.squeeze()

model = TelemetryLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 5. Modelltraining auf nominalen Daten
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Training des LSTM-Vorhersagemodells mit nominalen Telemetriedaten...")
model.train()
for epoch in range(EPOCHS):
    for bx, by in loader:
        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()
    print(f"Epoche [{epoch+1}/{EPOCHS}] abgeschlossen.")

# 6. Testphase und Berechnung des Vorhersageresiduums
model.eval()
with torch.no_grad():
    test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

# Absoluter Vorhersagefehler (Residuum): |Realer Wert - Vorhersage|
prediction_errors = np.abs(y_test - test_preds)

# 7. Fehlerglättung mittels EWMA (Hundman et al., 2018)
errors_smooth = pd.Series(prediction_errors).ewm(span=20).mean().values

# Dynamische Schwellenwertberechnung (mu + 3.0 * sigma)
threshold = np.mean(errors_smooth) + 3.0 * np.std(errors_smooth)
detected = errors_smooth > threshold

# 8. Visualisierung und Speicherung der Ergebnisse
df = pd.read_csv('labeled_anomalies.csv')
true_anomalies = ast.literal_eval(df[df['chan_id'] == 'P-1'].iloc[0]['anomaly_sequences'])

plt.figure(figsize=(14, 7))

# Oberer Plot: Reales Signal und LSTM-Vorhersage
plt.subplot(2, 1, 1)
plt.plot(test_raw, label='Reales Telemetriesignal', color='tab:blue', alpha=0.6)
time_axis = np.arange(WINDOW_SIZE, len(test_raw))
plt.plot(time_axis, test_preds, label='LSTM-Vorhersage (t+1)', color='darkgreen', alpha=0.7, linestyle=':')
for start, end in true_anomalies:
    plt.axvspan(start, end, color='red', alpha=0.3, label='NASA Ground Truth (Reale Anomalie)')
plt.title('1. SMAP P-1-Telemetrie und Ein-Schritt-Voraus-Vorhersage mittels LSTM')
plt.ylabel('Normalisierter Wert [-1, 1]')
plt.grid(True, linestyle='--', alpha=0.4)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper right')

# Unterer Plot: Residuum und FDIR-Erkennung
plt.subplot(2, 1, 2)
plt.plot(time_axis, errors_smooth, label='Vorhersageresiduum (EWMA)', color='darkorange', linewidth=1.2)
plt.axhline(threshold, color='red', linestyle='--', label=f'Dynamischer Schwellenwert ({threshold:.4f})')
plt.fill_between(time_axis, 0, errors_smooth, where=detected, color='purple', alpha=0.4, label='Autonomer FDIR-Alarm')

plt.title('2. Geglättetes Vorhersageresiduum (EWMA) und autonome Fehlererkennung')
plt.xlabel('Zeitschritt (Minuten)')
plt.ylabel('Residuum (|Realer Wert - Vorhersage|)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right')

plt.tight_layout()
# Grafiği rapor için yüksek çözünürlükte kaydeder
plt.savefig('Abbildung_3_LSTM_Ergebnisse.png', dpi=300)
plt.show()