"""
RUN TRAINING PSO-RANDOM FOREST
Jalankan file ini untuk training model PSO-RF
"""

import sys
import os

# Tambah path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🚀 STARTING PSO-RANDOM FOREST TRAINING")
print("=" * 70)

try:
    from app.train_pso_rf import train_pso_rf_model
    model_path = train_pso_rf_model()
    
    print("\n" + "=" * 70)
    print(f"✅ TRAINING SUCCESSFUL!")
    print(f"📁 Model saved at: {model_path}")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ TRAINING FAILED: {e}")
    import traceback
    traceback.print_exc()