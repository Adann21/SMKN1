#connection
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st

def get_connection():
    try:
        conn = psycopg2.connect(
            dbname="prediksi_db",
            user="postgres", 
            password="232102582",
            host="localhost",
            port="5432",
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        print(f"Database connection error: {e}")
        return None