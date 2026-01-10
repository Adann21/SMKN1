import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class PSORFOptimizer:
    """PSO-RF OPTIMIZER KOMPLEKS TANPA KOMPROMI"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.best_params = None
        self.feature_importance = None
        self.performance_metrics = {}
        
    def generate_synthetic_data(self, n_samples=15000):
        """Generate data kompleks dengan multiple clusters"""
        np.random.seed(42)
        data = []
        
        # CLUSTER 1: SISWA BERPRESTASI TINGGI (25%)
        n1 = int(n_samples * 0.25)
        for _ in range(n1):
            cluster = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            if cluster == 1:  # Prestasi akademik
                nilai = np.random.normal(92, 3)
                absensi = np.random.normal(96, 2)
                pelanggaran = np.random.poisson(0.3)
            elif cluster == 2:  # Prestasi ekstra
                nilai = np.random.normal(88, 4)
                absensi = np.random.normal(94, 3)
                pelanggaran = np.random.poisson(0.8)
            else:  # Prestasi seimbang
                nilai = np.random.normal(85, 5)
                absensi = np.random.normal(90, 4)
                pelanggaran = np.random.poisson(1.5)
            data.append([np.clip(nilai, 70, 100), np.clip(absensi, 85, 100), 
                        np.clip(pelanggaran, 0, 5), 1])
        
        # CLUSTER 2: SISWA BAIK (30%)
        n2 = int(n_samples * 0.30)
        for _ in range(n2):
            cluster = np.random.choice([1, 2], p=[0.6, 0.4])
            if cluster == 1:  # Baik akademik
                nilai = np.random.normal(82, 5)
                absensi = np.random.normal(88, 4)
                pelanggaran = np.random.poisson(2.5)
            else:  # Baik kedisiplinan
                nilai = np.random.normal(78, 6)
                absensi = np.random.normal(92, 3)
                pelanggaran = np.random.poisson(1.8)
            data.append([np.clip(nilai, 60, 95), np.clip(absensi, 75, 100), 
                        np.clip(pelanggaran, 0, 8), 1])
        
        # CLUSTER 3: SISWA CUKUP (20%)
        n3 = int(n_samples * 0.20)
        for _ in range(n3):
            nilai = np.random.normal(73, 7)
            absensi = np.random.normal(82, 6)
            pelanggaran = np.random.poisson(5.5)
            data.append([np.clip(nilai, 50, 90), np.clip(absensi, 65, 95), 
                        np.clip(pelanggaran, 0, 12), np.random.choice([0, 1], p=[0.3, 0.7])])
        
        # CLUSTER 4: SISWA BERMASALAH (25%)
        n4 = n_samples - (n1 + n2 + n3)
        for _ in range(n4):
            cluster = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            if cluster == 1:  # Masalah akademik
                nilai = np.random.normal(58, 10)
                absensi = np.random.normal(75, 8)
                pelanggaran = np.random.poisson(9)
            elif cluster == 2:  # Masalah disiplin
                nilai = np.random.normal(65, 8)
                absensi = np.random.normal(68, 10)
                pelanggaran = np.random.poisson(15)
            else:  # Masalah kombinasi
                nilai = np.random.normal(48, 12)
                absensi = np.random.normal(62, 12)
                pelanggaran = np.random.poisson(22)
            data.append([np.clip(nilai, 30, 75), np.clip(absensi, 50, 85), 
                        np.clip(pelanggaran, 5, 30), 0])
        
        df = pd.DataFrame(data, columns=['nilai', 'absensi', 'pelanggaran', 'label'])
        return df
    
    def pso_optimization(self, X, y, n_particles=20, max_iter=40):
        """PSO KOMPLEKS DENGAN MULTI-OBJECTIVE"""
        print("🚀 MEMULAI PSO OPTIMIZATION KOMPLEKS...")
        
        n_dim = 5  # n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
        
        bounds = np.array([
            [100, 500],    # n_estimators
            [5, 30],       # max_depth
            [2, 20],       # min_samples_split
            [1, 10],       # min_samples_leaf
            [0.1, 1.0]     # max_features (ratio)
        ])
        
        # Inisialisasi partikel
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, n_dim))
        velocities = np.zeros_like(particles)
        
        # Personal dan global best
        personal_best = particles.copy()
        personal_best_scores = np.full(n_particles, np.inf)
        global_best = particles[0].copy()
        global_best_score = np.inf
        
        # Stratified K-Fold untuk validasi
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Fungsi fitness multi-objective
        def evaluate_particle(params):
            try:
                n_est = int(params[0])
                max_d = int(params[1]) if params[1] > 5 else None
                min_split = int(params[2])
                min_leaf = int(params[3])
                max_feat = params[4]
                
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    max_features=max_feat,
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True,
                    oob_score=True
                )
                
                # Cross-validation metrics
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
                accuracy = np.mean(scores)
                
                # Complexity penalty
                complexity = (n_est/500) + (0 if max_d is None else max_d/30)
                
                # Multi-objective: maximize accuracy, minimize complexity
                fitness = (1 - accuracy) + 0.1 * complexity
                
                return fitness, accuracy
            except:
                return np.inf, 0
        
        # Evaluasi awal
        print("📊 EVALUASI PARTIKEL AWAL...")
        for i in range(n_particles):
            fitness, accuracy = evaluate_particle(particles[i])
            personal_best_scores[i] = fitness
            if fitness < global_best_score:
                global_best_score = fitness
                global_best = particles[i].copy()
                print(f"   Particle {i+1}: Accuracy = {accuracy:.4f}")
        
        # PSO MAIN LOOP DENGAN ADAPTIVE PARAMETERS
        print("\n🔄 PROSES OPTIMIZATION PSO...")
        w_min, w_max = 0.4, 0.9
        c1, c2 = 2.0, 2.0
        
        for iteration in range(max_iter):
            # Adaptive inertia weight
            w = w_max - (w_max - w_min) * (iteration / max_iter)
            
            for i in range(n_particles):
                # Update velocity dengan constriction factor
                r1 = np.random.rand(n_dim)
                r2 = np.random.rand(n_dim)
                
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
                
                # Evaluate
                fitness, accuracy = evaluate_particle(particles[i])
                
                # Update best
                if fitness < personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = fitness
                    
                    if fitness < global_best_score:
                        global_best = particles[i].copy()
                        global_best_score = fitness
            
            if (iteration + 1) % 10 == 0:
                _, best_acc = evaluate_particle(global_best)
                print(f"   Iteration {iteration+1}: Best Accuracy = {best_acc:.4f}")
        
        return global_best, global_best_score
    
    def train_complex_model(self):
        """TRAIN MODEL KOMPLEKS DENGAN OPTIMASI MAXIMAL"""
        print("=" * 80)
        print("🧠 PSO-RANDOM FOREST MODEL KOMPLEKS")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. GENERATE DATA KOMPLEKS
        print("\n📈 GENERATING COMPLEX SYNTHETIC DATA...")
        df = self.generate_synthetic_data(15000)
        print(f"✅ Data generated: {len(df):,} samples")
        print(f"   Distribution: Lulus={df['label'].sum():,}, Tidak={len(df)-df['label'].sum():,}")
        
        # 2. PREPROCESSING KOMPLEKS
        X = df[['nilai', 'absensi', 'pelanggaran']].values
        y = df['label'].values
        
        # Robust scaling untuk handle outliers
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 3. PSO OPTIMIZATION
        best_params, best_score = self.pso_optimization(X_scaled, y)
        
        # 4. TRAIN FINAL MODEL DENGAN BEST PARAMS
        print("\n🔥 TRAINING FINAL MODEL DENGAN PARAMETER OPTIMAL...")
        
        n_estimators = int(best_params[0])
        max_depth = int(best_params[1]) if best_params[1] > 5 else None
        min_samples_split = int(best_params[2])
        min_samples_leaf = int(best_params[3])
        max_features = best_params[4]
        
        self.best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }
        
        # Model dengan parameter optimal
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            min_impurity_decrease=0.0001,
            max_samples=0.8 if len(X) > 5000 else None
        )
        
        # Split data untuk validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Training
        self.model.fit(X_train, y_train)
        
        # 5. COMPREHENSIVE EVALUATION
        print("\n📊 EVALUASI MODEL KOMPREHENSIF...")
        
        # Predictions
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = self.model.score(X_val, y_val)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        oob_score = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Store metrics
        self.performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'oob_score': oob_score,
            'cv_score': 1 - best_score
        }
        
        # 6. PRINT RESULTS
        print("=" * 60)
        print("🎯 HASIL OPTIMASI PSO-RF")
        print("=" * 60)
        
        print(f"\n⚙️ PARAMETER OPTIMAL:")
        for key, value in self.best_params.items():
            print(f"   {key:20}: {value}")
        
        print(f"\n📈 METRIK KINERJA:")
        print(f"   Accuracy           : {accuracy:.4f}")
        print(f"   Precision          : {precision:.4f}")
        print(f"   Recall             : {recall:.4f}")
        print(f"   F1-Score           : {f1:.4f}")
        print(f"   ROC-AUC            : {roc_auc:.4f}")
        if oob_score:
            print(f"   OOB Score          : {oob_score:.4f}")
        
        print(f"\n📊 FEATURE IMPORTANCE:")
        features = ['Nilai Akademik', 'Kehadiran', 'Pelanggaran']
        for feat, imp in zip(features, self.feature_importance):
            print(f"   {feat:20}: {imp:.4f}")
        
        # 7. SAVE MODEL
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance.tolist(),
            'training_data_info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'class_distribution': {'lulus': int(y.sum()), 'tidak_lulus': len(y)-int(y.sum())},
                'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_version': 'complex_v1.0'
            }
        }
        
        model_path = os.path.join(models_dir, "pso_rf_complex.joblib")
        joblib.dump(model_data, model_path, compress=9)  # Max compression
        
        training_time = time.time() - start_time
        
        print(f"\n💾 MODEL DISIMPAN: {model_path}")
        print(f"📦 Size: {os.path.getsize(model_path) / 1024:.2f} KB")
        print(f"⏱️  Total Time: {training_time:.2f} seconds")
        
        # 8. TEST PREDICTIONS
        print("\n🧪 TEST PREDICTION:")
        print("-" * 50)
        
        test_cases = [
            [95, 98, 1, "BERPRESTASI TINGGI"],
            [85, 90, 3, "BAIK"],
            [75, 80, 8, "CUKUP"],
            [65, 70, 15, "BERMASALAH RINGAN"],
            [55, 60, 25, "BERMASALAH BERAT"]
        ]
        
        for nilai, absensi, pelanggaran, label in test_cases:
            X_test = np.array([[nilai, absensi, pelanggaran]])
            X_test_scaled = self.scaler.transform(X_test)
            
            pred = self.model.predict(X_test_scaled)[0]
            proba = self.model.predict_proba(X_test_scaled)[0]
            
            result = "✅ LULUS" if pred == 1 else "❌ TIDAK LULUS"
            confidence = max(proba) * 100
            
            print(f"\n{label}:")
            print(f"  Input: Nilai={nilai}, Absensi={absensi}%, Pelanggaran={pelanggaran}")
            print(f"  Prediksi: {result}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Probabilitas: Lulus={proba[1]*100:.1f}%, Tidak={proba[0]*100:.1f}%")
        
        print("\n" + "=" * 80)
        print("✅ TRAINING MODEL KOMPLEKS SELESAI!")
        print("=" * 80)
        
        return model_path

def train_pso_rf_model():
    """Fungsi utama untuk training"""
    optimizer = PSORFOptimizer()
    return optimizer.train_complex_model()

if __name__ == "__main__":
    train_pso_rf_model()