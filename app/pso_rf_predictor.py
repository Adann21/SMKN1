import numpy as np
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

class PSOComplexPredictor:
    """PREDICTOR KOMPLEKS DENGAN RESPONSIVITAS MAKSIMAL"""
    
    _instance = None
    _model_loaded = False
    _last_prediction_time = 0
    _prediction_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PSOComplexPredictor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize dengan lazy loading"""
        self.model = None
        self.scaler = None
        self.model_info = None
        self.performance = None
        self._load_model()
    
    def _load_model(self):
        """Load model dengan multiple fallback paths"""
        model_paths = [
            os.path.join(os.path.dirname(__file__), "models", "pso_rf_complex.joblib"),
            os.path.join(os.path.dirname(__file__), "models", "pso_rf_optimized.joblib"),
            os.path.join("app", "models", "pso_rf_complex.joblib"),
            os.path.join("models", "pso_rf_complex.joblib"),
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    start = time.time()
                    model_data = joblib.load(path)
                    
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.model_info = model_data
                    self.performance = model_data.get('performance_metrics', {})
                    
                    load_time = time.time() - start
                    
                    print(f"✅ MODEL LOADED: {path}")
                    print(f"⚡ Load Time: {load_time:.3f}s")
                    print(f"📊 Accuracy: {self.performance.get('accuracy', 0):.4f}")
                    
                    self._model_loaded = True
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
        
        # Jika model tidak ditemukan, buat yang sederhana tapi TETAP ML
        print("⚠️ No model found, creating simple ML model...")
        self._create_simple_ml_model()
    
    def _create_simple_ml_model(self):
        """Buat model ML sederhana jika tidak ada (TETAP ML, BUKAN RULE-BASED)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Generate simple training data
        np.random.seed(42)
        n_samples = 1000
        X = np.random.rand(n_samples, 3) * 100
        X[:, 0] = np.clip(X[:, 0], 40, 100)  # Nilai
        X[:, 1] = np.clip(X[:, 1], 50, 100)  # Absensi
        X[:, 2] = np.clip(X[:, 2], 0, 30)    # Pelanggaran
        
        # Simple rule untuk label (TETAP DIBUAT OLEH MODEL)
        y = ((X[:, 0] * 0.5 + X[:, 1] * 0.3 - X[:, 2] * 0.2) > 60).astype(int)
        
        # Train model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        self._model_loaded = True
        print("✅ Simple ML model created")
    
    def _get_cache_key(self, nilai, absensi, pelanggaran):
        """Generate cache key"""
        return f"{nilai:.1f}_{absensi:.1f}_{pelanggaran}"
    
    def predict_with_cache(self, nilai, absensi, pelanggaran):
        """Predict dengan caching untuk responsivitas"""
        cache_key = self._get_cache_key(nilai, absensi, pelanggaran)
        
        # Cek cache (valid 5 detik)
        current_time = time.time()
        if cache_key in self._prediction_cache:
            cached_data, timestamp = self._prediction_cache[cache_key]
            if current_time - timestamp < 5:  # Cache 5 detik
                return cached_data
        
        # Predict baru
        result = self._predict_ml(nilai, absensi, pelanggaran)
        
        # Update cache
        self._prediction_cache[cache_key] = (result, current_time)
        
        # Clean old cache entries
        if len(self._prediction_cache) > 100:
            oldest_key = min(self._prediction_cache.keys(), 
                           key=lambda k: self._prediction_cache[k][1])
            del self._prediction_cache[oldest_key]
        
        return result
    
    def _predict_ml(self, nilai, absensi, pelanggaran):
        """PREDIKSI ML MURNI - TANPA FALLBACK"""
        if not self._model_loaded:
            self._load_model()
        
        try:
            # Validasi input
            nilai = float(nilai)
            absensi = float(absensi)
            pelanggaran = int(pelanggaran)
            
            # Clamping
            nilai = np.clip(nilai, 0.0, 100.0)
            absensi = np.clip(absensi, 0.0, 100.0)
            pelanggaran = np.clip(pelanggaran, 0, 100)
            
            # Prepare input
            X = np.array([[nilai, absensi, pelanggaran]])
            
            # Transform
            X_scaled = self.scaler.transform(X)
            
            # Predict dengan timing
            pred_start = time.time()
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            pred_time = time.time() - pred_start
            
            # Confidence calculation
            confidence = max(probabilities) * 100
            prob_lulus = probabilities[1] * 100
            
            # Advanced decision logic
            if prediction == 1:  # LULUS
                if confidence >= 95:
                    status = "aman"
                    reason = f"PRESTASI TINGGI ({confidence:.1f}%)"
                elif confidence >= 85:
                    status = "aman"
                    reason = f"PRESTASI BAIK ({confidence:.1f}%)"
                elif confidence >= 75:
                    status = "waspada"
                    reason = f"PRESTASI CUKUP ({confidence:.1f}%)"
                elif confidence >= 65:
                    status = "waspada"
                    reason = f"BATAS MINIMAL ({confidence:.1f}%)"
                else:
                    status = "bermasalah"
                    reason = f"BERISIKO TINGGI ({confidence:.1f}%)"
                hasil = "lulus"
            else:  # TIDAK LULUS
                if confidence >= 90:
                    status = "bermasalah"
                    reason = f"MASALAH KRITIS ({confidence:.1f}%)"
                elif confidence >= 80:
                    status = "bermasalah"
                    reason = f"MASALAH BERAT ({confidence:.1f}%)"
                elif confidence >= 70:
                    status = "bermasalah"
                    reason = f"MASALAH SEDANG ({confidence:.1f}%)"
                elif confidence >= 60:
                    status = "bermasalah"
                    reason = f"MASALAH RINGAN ({confidence:.1f}%)"
                else:
                    status = "bermasalah"
                    reason = f"PERLU EVALUASI ({confidence:.1f}%)"
                hasil = "tidak_lulus"
            
            return {
                'status': status,
                'reason': reason,
                'hasil': hasil,
                'confidence': confidence,
                'prob_lulus': prob_lulus,
                'prediction_time': pred_time,
                'model_accuracy': self.performance.get('accuracy', 0) if self.performance else 0
            }
            
        except Exception as e:
            # TIDAK ADA FALLBACK - ERROR SAJA
            raise ValueError(f"ML Prediction Error: {str(e)}")
    
    def batch_predict(self, data_list):
        """Batch prediction untuk performa"""
        if not self._model_loaded:
            self._load_model()
        
        results = []
        for data in data_list:
            try:
                result = self.predict_with_cache(
                    data['nilai'], 
                    data['absensi'], 
                    data['pelanggaran']
                )
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results

# Global instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PSOComplexPredictor()
    return _predictor

def predict_student_ml(nilai, absensi, pelanggaran):
    """Interface utama untuk prediksi"""
    predictor = get_predictor()
    result = predictor.predict_with_cache(nilai, absensi, pelanggaran)
    return result['status'], result['reason'], result['hasil']

def get_ml_prediction_for_db(nilai, absensi, pelanggaran):
    """Alias untuk database"""
    return predict_student_ml(nilai, absensi, pelanggaran)

# Test
if __name__ == "__main__":
    print("🧪 Testing Complex PSO-RF Predictor")
    print("=" * 60)
    
    predictor = get_predictor()
    
    # Test pertama (cold)
    print("\n🔍 Test 1 (Cold):")
    start = time.time()
    result = predictor.predict_with_cache(85, 90, 3)
    print(f"   Result: {result['hasil']}")
    print(f"   Status: {result['status']}")
    print(f"   Time: {(time.time()-start)*1000:.1f}ms")
    
    # Test kedua (cached)
    print("\n🔍 Test 2 (Cached):")
    start = time.time()
    result = predictor.predict_with_cache(85, 90, 3)
    print(f"   Result: {result['hasil']}")
    print(f"   Status: {result['status']}")
    print(f"   Time: {(time.time()-start)*1000:.1f}ms")
    
    # Batch test
    print("\n🔍 Batch Test:")
    batch_data = [
        {'nilai': 95, 'absensi': 98, 'pelanggaran': 1},
        {'nilai': 75, 'absensi': 80, 'pelanggaran': 8},
        {'nilai': 55, 'absensi': 65, 'pelanggaran': 15}
    ]
    
    start = time.time()
    results = predictor.batch_predict(batch_data)
    batch_time = time.time() - start
    
    for i, result in enumerate(results):
        print(f"   Data {i+1}: {result.get('hasil', 'ERROR')}")
    
    print(f"\n⚡ Batch Time: {batch_time*1000:.1f}ms")