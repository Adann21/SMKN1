import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("🤖 CREATING PSO-RF MODEL FOR STUDENT GRADUATION")
print("=" * 60)

# Buat folder models
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# Panggil training PSO-RF
try:
    from app.train_pso_rf import train_pso_rf_model
    model_path = train_pso_rf_model()
    print(f"\n✅ PSO-RF model created: {model_path}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    
    # Buat model sederhana sebagai fallback
    print("Creating simple RF model as fallback...")
    
    # Data training sederhana
    training_data = [
        [90, 95, 0, 1],
        [85, 90, 1, 1],
        [80, 85, 2, 1],
        [75, 80, 3, 1],
        [70, 75, 4, 1],
        [65, 70, 5, 0],
        [60, 65, 6, 0],
        [55, 60, 7, 0],
        [50, 55, 8, 0],
        [45, 50, 9, 0],
    ]
    
    df = pd.DataFrame(training_data, columns=['nilai', 'absensi', 'pelanggaran', 'label'])
    X = df[['nilai', 'absensi', 'pelanggaran']].values
    y = df['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    model_path = os.path.join(models_dir, "pso_rf_model_light.joblib")
    joblib.dump(model_data, model_path)
    print(f"💾 Simple model saved: {model_path}")

print("=" * 60)
print("🎯 MODEL CREATION PROCESS COMPLETED")
print("=" * 60)