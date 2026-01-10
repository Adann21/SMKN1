# ⭐⭐⭐ HANYA DATABASE OPERATIONS ⭐⭐⭐
from database.connection import get_connection
import sys
import os

# ========== FIX IMPORT ML ANJING ==========
# Fix path buat import dari app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
app_dir = os.path.join(parent_dir, 'app')

# Tambah ke sys.path
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from pso_rf_predictor import get_ml_prediction_for_db
    print("✅ PSO-RF predictor loaded successfully from app")
except ImportError as e:
    print(f"❌ Error importing PSO-RF: {e}")
    
    # Coba import langsung dari path
    try:
        predictor_path = os.path.join(app_dir, 'pso_rf_predictor.py')
        if os.path.exists(predictor_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("pso_rf_predictor", predictor_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_ml_prediction_for_db = module.get_ml_prediction_for_db
            print("✅ PSO-RF predictor loaded directly from file")
        else:
            raise ImportError(f"File not found: {predictor_path}")
    except Exception as e2:
        print(f"❌ All imports failed: {e2}")
        # Fallback buatan sendiri
        def get_ml_prediction_for_db(nilai, absensi, pelanggaran):
            print("⚠️ Using fallback prediction")
            score = (nilai * 0.6) + (absensi * 0.3) - (pelanggaran * 5)
            if score >= 70:
                return "aman", f"Fallback: Baik ({score:.1f})", "lulus"
            elif score >= 60:
                return "waspada", f"Fallback: Cukup ({score:.1f})", "lulus"
            elif score >= 50:
                return "waspada", f"Fallback: Batas ({score:.1f})", "tidak_lulus"
            else:
                return "bermasalah", f"Fallback: Rendah ({score:.1f})", "tidak_lulus"

# ========== FUNGSI DATABASE ==========
def get_students(search=None):
    """Mendapatkan semua data siswa dari database"""
    print(f"📊 GET STUDENTS dipanggil dengan search: {search}")
    
    conn = get_connection()
    if not conn:
        print("❌ Koneksi database gagal")
        return []
    
    try:
        cur = conn.cursor()
        
        if search and search.strip():
            q = f"%{search.strip()}%"
            cur.execute("""
                SELECT id, nis, nama, kelas, jurusan, nilai_rata_rata, 
                absensi, pelanggaran, tahun_angkatan, status, 
                peringatan, prediksi_alasan, prediksi_hasil,
                prediksi_tanggal, created_at
                FROM students 
                WHERE nis ILIKE %s OR nama ILIKE %s OR kelas ILIKE %s OR jurusan ILIKE %s
                ORDER BY created_at DESC
            """, (q, q, q, q))
        else:
            cur.execute("""
                SELECT id, nis, nama, kelas, jurusan, nilai_rata_rata, 
                absensi, pelanggaran, tahun_angkatan, status, 
                peringatan, prediksi_alasan, prediksi_hasil,
                prediksi_tanggal, created_at
                FROM students 
                ORDER BY created_at DESC
            """)
        
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            results.append(dict(row))
        
        print(f"✅ Data siswa ditemukan: {len(results)} records")
        return results
        
    except Exception as e:
        print(f"❌ Error get_students: {e}")
        return []
    finally:
        if conn: 
            conn.close()
def create_student(data: dict):
    """Membuat siswa baru dengan 100% ML prediction"""
    print(f"🎯 CREATE STUDENT 100% ML: {data.get('nis')}")
    
    conn = get_connection()
    if not conn: 
        return False, "Koneksi database gagal"
    
    try:
        cur = conn.cursor()
        
        # 1. CEK NIS - SATU KALI SAJA
        nis = data.get("nis", "").strip()
        
        # Cek duplikasi
        cur.execute("SELECT id FROM students WHERE nis = %s", (nis,))
        if cur.fetchone(): 
            return False, f"NIS {nis} sudah digunakan"
        
        # 2. Validasi input
        nilai_rata_rata = float(data.get("nilai_rata_rata", 0))
        absensi_val = float(data.get("absensi", 0))
        pelanggaran_val = int(data.get("pelanggaran", 0))
        
        # 3. ⭐⭐⭐ PREDIKSI PSO-RF ⭐⭐⭐
        print(f"🤖 Running PSO-RF prediction for {nis}...")
        peringatan, prediksi_alasan, prediksi_hasil = get_ml_prediction_for_db(
            nilai_rata_rata, absensi_val, pelanggaran_val
        )
        
        # 4. INSERT SEKALI SAJA
        query = """
            INSERT INTO students (
                nis, nama, kelas, jurusan, nilai_rata_rata, absensi, 
                pelanggaran, tahun_angkatan, status,
                peringatan, prediksi_alasan, prediksi_hasil, prediksi_tanggal
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE) 
            RETURNING id
        """
        params = (
            nis,
            data.get("nama", "").strip(),
            data.get("kelas", "").strip(),
            data.get("jurusan", "").strip(),
            nilai_rata_rata,
            absensi_val,
            pelanggaran_val,
            int(data.get("tahun_angkatan", 2024)),
            "aktif",
            peringatan,
            prediksi_alasan,
            prediksi_hasil
        )
        
        cur.execute(query, params)
        new_id = cur.fetchone()['id']
        conn.commit()
        
        print(f"✅ Student created: {nis} (ID: {new_id})")
        return True, new_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        return False, f"Error: {str(e)}"
    finally:
        if conn: 
            conn.close()

def delete_student(student_id):
    """Hapus data siswa"""
    conn = get_connection()
    if not conn:
        return False, "Koneksi database gagal"
    
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM students WHERE id = %s", (student_id,))
        conn.commit()
        return True, f"Siswa dengan ID {student_id} berhasil dihapus"
    except Exception as e:
        conn.rollback()
        return False, f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()

def delete_all_students():
    """Hapus semua data siswa"""
    conn = get_connection()
    if not conn:
        return False, "Koneksi database gagal"
    
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM students")
        cur.execute("ALTER SEQUENCE students_id_seq RESTART WITH 1")
        conn.commit()
        return True, "Semua data siswa berhasil dihapus"
    except Exception as e:
        conn.rollback()
        return False, f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()

def update_student(student_id, data: dict):
    """Update data siswa dengan 100% ML"""
    conn = get_connection()
    if not conn:
        return False, "Koneksi database gagal"
    
    try:
        cur = conn.cursor()
        
        nilai_rata_rata = float(data.get("nilai_rata_rata", 0))
        absensi_val = float(data.get("absensi", 0))
        pelanggaran_val = int(data.get("pelanggaran", 0))
        
        nilai_rata_rata = max(0.0, min(100.0, nilai_rata_rata))
        absensi_val = max(0.0, min(100.0, absensi_val))
        
        peringatan, prediksi_alasan, prediksi_hasil = get_ml_prediction_for_db(
            nilai_rata_rata, absensi_val, pelanggaran_val
        )
        
        query = """
            UPDATE students SET
                nis = %s,
                nama = %s,
                kelas = %s,
                jurusan = %s,
                nilai_rata_rata = %s,
                absensi = %s,
                pelanggaran = %s,
                tahun_angkatan = %s,
                status = %s,
                peringatan = %s,
                prediksi_alasan = %s,
                prediksi_hasil = %s,
                prediksi_tanggal = CURRENT_DATE
            WHERE id = %s
        """
        
        params = (
            data.get("nis", "").strip(),
            data.get("nama", "").strip(),
            data.get("kelas", "").strip(),
            data.get("jurusan", "").strip(),
            nilai_rata_rata,
            absensi_val,
            pelanggaran_val,
            int(data.get("tahun_angkatan", 2024)),
            data.get("status", "aktif"),
            peringatan,
            prediksi_alasan,
            prediksi_hasil,
            student_id
        )
        
        cur.execute(query, params)
        conn.commit()
        
        return True, "Data siswa berhasil diperbarui dengan ML"
        
    except Exception as e:
        conn.rollback()
        return False, f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()