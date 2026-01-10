import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Kelulusan",
    page_icon="🎓",
    layout="wide"
)

# Hilangkan watermark Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    #stDecoration {display: none;}
    .stToolbar {display: none !important;}
    
    /* Sembunyikan hamburger menu */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Import dashboard admin
from app.admin_tu import dashboard as admin_dashboard

def login_page():
    """Halaman login SEDERHANA"""
    st.title("🎓 SMKN 1 Sandai")
    st.subheader("Sistem Prediksi Kelulusan Siswa")
    
    st.markdown("---")
    
    # Login sederhana - hanya 1 admin
    USERNAME = "admin_sandai"
    PASSWORD = "sandai2024"  # Ganti dengan password Anda
    
    # Form login
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("LOGIN", use_container_width=True)
    
    # Cek login
    if submitted:
        if username == USERNAME and password == PASSWORD:
            # INISIALISASI SESSION STATE dengan benar
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = "admin"
            st.success("✅ Login berhasil!")
            st.rerun()  # Gunakan st.rerun() untuk refresh
        else:
            st.error("❌ Username atau password salah!")
    
    # Info footer
    st.markdown("---")
    st.caption("⚠️ Hanya untuk Admin SMKN 1 Sandai")

# INISIALISASI SESSION STATE JIKA BELUM ADA
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""

# Routing dengan pengecekan yang benar
if not st.session_state.logged_in:  # Sekarang aman diakses
    login_page()
else:
    admin_dashboard()