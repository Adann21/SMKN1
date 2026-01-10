# ⭐⭐⭐ HANYA DASHBOARD STREAMLIT ⭐⭐⭐
import streamlit as st
import pandas as pd
from database import students_crud as crud
from io import BytesIO
import sys
import os

# ⭐⭐⭐ IMPORT ML UNTUK DASHBOARD SAJA ⭐⭐⭐
try:
    from app.pso_rf_predictor import predict_student_ml
    print("✅ ML module imported successfully")
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

def ensure_admin():
    """Pastikan hanya admin yang boleh masuk."""
    if not st.session_state.get("logged_in"):
        st.error("Akses ditolak. Silakan login terlebih dahulu.")
        st.stop()

def dashboard():
    ensure_admin()

    # --- CSS STYLING ---
    st.markdown("""
        <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {display:none;}
        [data-testid="stDecoration"] {display:none;}
        [data-testid="collapsedControl"] {display: block !important;}
        .css-1d391kg {display: block !important;}
        .css-1cypcdb {display: block !important;}
        
        .sidebar-title {
            font-size: 18px;
            font-weight: bold;
            text-align: left;
            margin-bottom: 10px;
            padding-left: 10px;
        }
        
        .stButton button {
            text-align: left !important;
            justify-content: flex-start !important;
            padding-left: 15px !important;
        }
        
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        
        .main {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }
        
        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .full-container {
            width: 100% !important;
            padding: 0px 20px 20px 20px !important;
        }
        
        .ml-badge {
            background-color: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # --------------------------------------
    # SIDEBAR MENU
    # --------------------------------------
    st.sidebar.markdown("<div class='sidebar-title'>⚙️ Menu Admin</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    if 'menu' not in st.session_state:
        st.session_state.menu = "Dashboard"

    menu_items = [
        "🏠 Dashboard",
        "➕ Tambah Siswa",
        "📋 Daftar Siswa",
        "🔮 Prediksi Siswa"
    ]

    for item in menu_items:
        if st.sidebar.button(item, use_container_width=True, key=f"menu_{item}"):
            st.session_state.menu = item.split(" ", 1)[1]

    # Info user
    st.sidebar.markdown("---")
    st.sidebar.info(f"👤 **Admin**\n\nSMKN 1 Sandai")
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # HEADER
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.markdown("<div class='main-title'>📌 Dashboard Admin SMKN 1 Sandai</div>", unsafe_allow_html=True)
    with col_header2:
        st.markdown('<div class="ml-badge">🤖 100% ML PSO-RF</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div class='full-container'>", unsafe_allow_html=True)

    # =====================================================
    # FUNGSI CACHE
    # =====================================================
    @st.cache_data(ttl=300)
    def get_all_students_cached():
        return crud.get_students("")

    # =====================================================
    # DASHBOARD - STATISTIK
    # =====================================================
    if st.session_state.menu == "Dashboard":
        st.subheader("📊 Dashboard Statistik")
        
        rows = get_all_students_cached()
        total_siswa = len(rows)
        aktif = len([r for r in rows if r.get("status") == "aktif"])
        lulus = len([r for r in rows if r.get("status") == "lulus"])
        dropout = len([r for r in rows if r.get("status") == "dropout"])
        
        aman_count = len([r for r in rows if r.get("peringatan") == "aman"])
        waspada_count = len([r for r in rows if r.get("peringatan") == "waspada"])
        bermasalah_count = len([r for r in rows if r.get("peringatan") == "bermasalah"])

        colA, colB, colC, colD = st.columns(4, gap="large")
        with colA: st.metric("Total", total_siswa)
        with colB: st.metric("Aktif", aktif)
        with colC: st.metric("Lulus", lulus)
        with colD: st.metric("Dropout", dropout)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("### 🚨 Status Peringatan Siswa")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("✅ AMAN", aman_count)
        with col2: st.metric("⚠️ WASPADA", waspada_count)
        with col3: st.metric("🔴 BERMASALAH", bermasalah_count)
        
        st.markdown("---")
        st.write("### 📈 Overview Sistem")
        st.info("Sistem Prediksi Kelulusan SMKN 1 Sandai - Powered by PSO-Random Forest")

    # =====================================================
    # TAMBAH SISWA
    # =====================================================
    elif st.session_state.menu == "Tambah Siswa":
        st.subheader("➕ Tambah Siswa Baru")
        
        tab1, tab2 = st.tabs(["📝 Input Manual", "📤 Import Excel"])
        
        with tab1:
            with st.form("tambah_siswa_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    nis = st.text_input("NIS *", placeholder="Contoh: 2024001")
                    nama = st.text_input("Nama Lengkap *", placeholder="Contoh: Ahmad Rizki")
                    nilai_rata_rata = st.number_input("Nilai Rata-rata *", min_value=0.0, max_value=100.0, value=0.0)
                with col2:
                    kelas = st.text_input("Kelas", placeholder="Contoh: X-TKJ-1")
                    jurusan = st.text_input("Jurusan", placeholder="Contoh: TKJ")
                    absensi = st.number_input("Kehadiran (%) *", min_value=0, max_value=100, value=0)
                
                pelanggaran = st.number_input("Pelanggaran Kedisiplinan (jumlah) *", min_value=0, max_value=100, value=0)
                tahun_angkatan = st.number_input("Tahun Angkatan *", min_value=2000, max_value=2100, value=2024)
                status = st.selectbox("Status Siswa *", ["aktif"], help="Untuk prediksi ML, status harus 'aktif'")
                
                submit_button = st.form_submit_button("💾 Simpan Siswa", use_container_width=True)
                
                if submit_button:
                    if not nis.strip() or not nama.strip():
                        st.error("❌ NIS dan Nama wajib diisi!")
                    else:
                        data = {
                            "nis": nis.strip(),
                            "nama": nama.strip(), 
                            "kelas": kelas.strip() if kelas else "",
                            "jurusan": jurusan.strip() if jurusan else "",
                            "nilai_rata_rata": float(nilai_rata_rata),
                            "absensi": float(absensi),
                            "pelanggaran": int(pelanggaran),
                            "tahun_angkatan": int(tahun_angkatan),
                            "status": status,
                        }
                        
                        ok, result = crud.create_student(data)
                        if ok:
                            st.success(f"✅ Berhasil menambah siswa {nama}!")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"❌ Gagal: {result}")
        
        with tab2:
            st.markdown("### 📥 Import dari Excel")
            st.markdown("#### 📋 Download Template")
            
            template_data = {
                "nis": ["12345", "12346", "12347"],
                "nama": ["Nama Siswa 1", "Nama Siswa 2", "Nama Siswa 3"],
                "kelas": ["X-TKJ-1", "X-TKJ-1", "XII-TKJ-1"],
                "jurusan": ["TKJ", "TKJ", "TKJ"],
                "nilai_rata_rata": [85.5, 78.0, 45.0],
                "absensi": [90, 85, 60],
                "pelanggaran": [2, 0, 20],
                "tahun_angkatan": [2024, 2024, 2023],
                "status": ["aktif", "aktif", "aktif"]
            }
            
            template_df = pd.DataFrame(template_data)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_template = template_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Template CSV", csv_template, "template_siswa.csv", "text/csv", use_container_width=True)
            with col_dl2:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    template_df.to_excel(writer, index=False, sheet_name='Template Siswa')
                st.download_button("📥 Download Template Excel", excel_buffer.getvalue(), "template_siswa.xlsx", 
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            
            st.info("💡 **Petunjuk:** Download template, isi data siswa, lalu upload file yang sudah diisi.")
            st.markdown("---")
            
            st.markdown("#### 📤 Upload File yang Sudah Diisi")
            uploaded_file = st.file_uploader("Pilih file Excel atau CSV", type=['xlsx', 'csv'])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_import = pd.read_csv(uploaded_file)
                    else:
                        df_import = pd.read_excel(uploaded_file)
                    
                    required_columns = ['nis', 'nama']
                    missing_columns = [col for col in required_columns if col not in df_import.columns]
                    
                    if missing_columns:
                        st.error(f"❌ Kolom wajib tidak ditemukan: {', '.join(missing_columns)}")
                    else:
                        st.markdown("#### 👀 Preview Data Input")
                        st.dataframe(df_import, use_container_width=True, height=300)
                        
                        if st.button("🚀 Import Data Siswa", use_container_width=True, type="primary"):
                            success_count = 0
                            error_count = 0
                            errors = []
                            
                            with st.spinner("Sedang mengimport data..."):
                                for index, row in df_import.iterrows():
                                    try:
                                        data = {
                                            "nis": str(row['nis']).strip(),
                                            "nama": str(row['nama']).strip(),
                                            "kelas": str(row.get('kelas', '')).strip(),
                                            "jurusan": str(row.get('jurusan', '')).strip(),
                                            "nilai_rata_rata": float(row.get('nilai_rata_rata', 0)),
                                            "absensi": int(row.get('absensi', 0)),
                                            "pelanggaran": int(row.get('pelanggaran', 0)),
                                            "tahun_angkatan": int(row.get('tahun_angkatan', 2024)),
                                            "status": "aktif"
                                        }
                                        
                                        if not data['nis'] or not data['nama']:
                                            error_count += 1
                                            errors.append(f"Baris {index+2}: NIS dan Nama wajib diisi")
                                            continue
                                        
                                        ok, result = crud.create_student(data)
                                        if ok: success_count += 1
                                        else:
                                            error_count += 1
                                            errors.append(f"Baris {index+2}: {result}")
                                            
                                    except Exception as e:
                                        error_count += 1
                                        errors.append(f"Baris {index+2}: Error - {str(e)}")
                            
                            if success_count > 0:
                                st.success(f"✅ Berhasil mengimport {success_count} data siswa!")
                                st.cache_data.clear()
                                
                            if error_count > 0:
                                st.error(f"❌ Gagal mengimport {error_count} data")
                                with st.expander("Detail Error"):
                                    for error in errors: st.write(f"- {error}")
                            
                except Exception as e:
                    st.error(f"❌ Error membaca file: {str(e)}")

    # =====================================================
    # DAFTAR SISWA - DENGAN FITUR HAPUS
    # =====================================================
    elif st.session_state.menu == "Daftar Siswa":
        st.subheader("📋 Daftar Siswa")
        
        # --- TOMBOL HAPUS SEMUA DI ATAS ---
        col_search, col_delete = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("🔍 Cari (NIS/Nama)")
        with col_delete:
            if st.button("🗑️ Hapus Semua", type="secondary", use_container_width=True):
                st.session_state['show_delete_all'] = True
        
        # --- KONFIRMASI HAPUS SEMUA ---
        if st.session_state.get('show_delete_all', False):
            st.error("⚠️ **PERINGATAN!** Anda akan menghapus SEMUA data siswa!")
            confirm = st.text_input("Ketik 'HAPUS-SEMUA' untuk konfirmasi:")
            
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("✅ YA, HAPUS SEMUA", type="primary", use_container_width=True):
                    if confirm == "HAPUS-SEMUA":
                        with st.spinner("Menghapus semua data siswa..."):
                            ok, message = crud.delete_all_students()
                            if ok:
                                st.success(f"✅ {message}")
                                st.cache_data.clear()
                                st.session_state.pop('show_delete_all', None)
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.error("Kode konfirmasi salah!")
            with col_confirm2:
                if st.button("❌ BATALKAN", type="secondary", use_container_width=True):
                    st.session_state.pop('show_delete_all', None)
                    st.rerun()
        
        # --- GET DATA ---
        rows = get_all_students_cached()
        
        if search_query:
            rows = [r for r in rows if search_query.lower() in str(r.get('nis', '')).lower() 
                    or search_query.lower() in str(r.get('nama', '')).lower()]
        
        if not rows:
            st.info("📭 Tidak ada data siswa.")
        else:
            display_data = []
            for row in rows:
                display_data.append({
                    'NIS': row.get('nis', ''),
                    'Nama': row.get('nama', ''),
                    'Kelas': row.get('kelas', ''),
                    'Jurusan': row.get('jurusan', ''),
                    'Nilai': row.get('nilai_rata_rata', 0),
                    'Absensi': row.get('absensi', 0),
                    'Pelanggaran': row.get('pelanggaran', 0),
                    'Tahun': row.get('tahun_angkatan', ''),
                    'Status': row.get('status', ''),
                    'ID': row.get('id', '')  # Tambahkan ID untuk hapus
                })
            
            df_display = pd.DataFrame(display_data)
            
            # Tampilkan dataframe tanpa kolom ID
            st.dataframe(df_display.drop(columns=['ID']), use_container_width=True, height=400)
            
            # --- FITUR HAPUS PER SISWA ---
            st.markdown("---")
            st.subheader("🗑️ Hapus Data Per Siswa")
            
            # Dropdown pilih siswa
            siswa_options = []
            for row in rows:
                siswa_id = row.get('id')
                if siswa_id:
                    siswa_options.append({
                        'label': f"{row.get('nis')} - {row.get('nama')}",
                        'value': siswa_id
                    })
            
            if siswa_options:
                selected_option = st.selectbox(
                    "Pilih siswa yang akan dihapus:",
                    options=siswa_options,
                    format_func=lambda x: x['label']
                )
                
                if selected_option:
                    # Tampilkan info siswa terpilih
                    selected_row = next((r for r in rows if r.get('id') == selected_option['value']), None)
                    if selected_row:
                        col_info, col_btn = st.columns([3, 1])
                        with col_info:
                            st.warning(f"""
                            **Data yang akan dihapus:**
                            - **NIS:** {selected_row.get('nis')}
                            - **Nama:** {selected_row.get('nama')}
                            - **Kelas:** {selected_row.get('kelas')}
                            - **Status:** {selected_row.get('status')}
                            """)
                        with col_btn:
                            if st.button("🚮 Hapus", type="primary", use_container_width=True):
                                st.session_state['delete_student_id'] = selected_option['value']
                                st.session_state['delete_student_name'] = selected_row.get('nama')
            
            # Konfirmasi hapus per siswa
            if st.session_state.get('delete_student_id'):
                st.error(f"⚠️ Konfirmasi Hapus: {st.session_state.get('delete_student_name')}")
                confirm_code = st.text_input("Ketik 'HAPUS' untuk melanjutkan:")
                
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ YA, HAPUS", type="primary", use_container_width=True):
                        if confirm_code == "HAPUS":
                            with st.spinner("Menghapus data siswa..."):
                                ok, message = crud.delete_student(st.session_state['delete_student_id'])
                                if ok:
                                    st.success(f"✅ {message}")
                                    st.cache_data.clear()
                                    st.session_state.pop('delete_student_id', None)
                                    st.session_state.pop('delete_student_name', None)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        else:
                            st.error("Ketik 'HAPUS' dengan benar!")
                with col_no:
                    if st.button("❌ BATALKAN", type="secondary", use_container_width=True):
                        st.session_state.pop('delete_student_id', None)
                        st.session_state.pop('delete_student_name', None)
                        st.rerun()
            
            # --- TOMBOL HAPUS CEPAT BERDASARKAN NIS ---
            with st.expander("⚡ Hapus Cepat (By NIS)", expanded=False):
                nis_to_delete = st.text_input("Masukkan NIS yang akan dihapus:")
                
                if nis_to_delete and st.button("🔍 Cari & Hapus", type="secondary"):
                    # Cari siswa berdasarkan NIS
                    siswa_ditemukan = next((r for r in rows if r.get('nis') == nis_to_delete), None)
                    if siswa_ditemukan:
                        st.warning(f"✅ Ditemukan: {siswa_ditemukan.get('nama')}")
                        
                        if st.button(f"🗑️ Hapus {nis_to_delete}", type="primary"):
                            ok, message = crud.delete_student(siswa_ditemukan.get('id'))
                            if ok:
                                st.success(f"✅ {message}")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.error(f"❌ NIS {nis_to_delete} tidak ditemukan!")

    # =====================================================
    # PREDIKSI SISWA
    # =====================================================
    elif st.session_state.menu == "Prediksi Siswa":
        st.subheader("🔮 Hasil Prediksi PSO-Random Forest")
        st.success("🤖 **Prediksi menggunakan PSO-Random Forest**")
        
        st.markdown("---")
        
        rows = get_all_students_cached()
        
        if not rows:
            st.info("📭 Tidak ada data siswa.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.selectbox("Filter Status:", ["Semua", "aktif", "lulus", "dropout"])
            with col2:
                prediksi_filter = st.selectbox("Filter Prediksi:", ["Semua", "lulus", "tidak_lulus"])
            
            filtered_rows = rows
            if status_filter != "Semua":
                filtered_rows = [r for r in filtered_rows if r.get("status") == status_filter]
            if prediksi_filter != "Semua":
                filtered_rows = [r for r in filtered_rows if r.get("prediksi_hasil") == prediksi_filter]
            
            total_filtered = len(filtered_rows)
            lulus_count = len([r for r in filtered_rows if r.get("prediksi_hasil") == "lulus"])
            tidak_lulus_count = len([r for r in filtered_rows if r.get("prediksi_hasil") == "tidak_lulus"])
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1: st.metric("Total", total_filtered)
            with col_stat2: st.metric("✅ LULUS", lulus_count)
            with col_stat3: st.metric("❌ TIDAK LULUS", tidak_lulus_count)
            
            st.markdown("---")
            
            if filtered_rows:
                result_data = []
                for row in filtered_rows:
                    prediksi_hasil = row.get('prediksi_hasil', '')
                    if prediksi_hasil == "lulus":
                        prediksi_icon = "✅"
                        prediksi_text = "LULUS"
                    elif prediksi_hasil == "tidak_lulus":
                        prediksi_icon = "❌"
                        prediksi_text = "TIDAK LULUS"
                    else:
                        prediksi_icon = "❓"
                        prediksi_text = "BELUM DIHITUNG"
                    
                    peringatan = row.get('peringatan', '')
                    peringatan_icon = {
                        "aman": "🟢",
                        "waspada": "🟡", 
                        "bermasalah": "🔴",
                    }.get(peringatan, "❓")
                    
                    result_data.append({
                        'NIS': row.get('nis', ''),
                        'Nama': row.get('nama', ''),
                        'Kelas': row.get('kelas', ''),
                        'Jurusan': row.get('jurusan', ''),
                        'Nilai': row.get('nilai_rata_rata', 0),
                        'Absensi': row.get('absensi', 0),
                        'Pelanggaran': row.get('pelanggaran', 0),
                        'Tahun': row.get('tahun_angkatan', ''),
                        'Status': row.get('status', ''),
                        'Peringatan': f"{peringatan_icon} {peringatan.upper()}" if peringatan else "❓",
                        'Prediksi': f"{prediksi_icon} {prediksi_text}"
                    })
                
                df_result = pd.DataFrame(result_data)
                st.dataframe(df_result, use_container_width=True, height=400)
                
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Hasil Prediksi", 
                    csv, 
                    "hasil_prediksi.csv", 
                    "text/csv", 
                    use_container_width=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ⭐⭐⭐ HANYA INI YANG ADA DI FILE admin_tu.py ⭐⭐⭐