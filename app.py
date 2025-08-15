import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import hashlib
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Database ---
DB_NAME = "database.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_usertable():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
              (username, hash_password(password)))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

# --- Session State ---
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_page(page_name):
    st.session_state.page = page_name

create_usertable()

# --- Home Page ---
if st.session_state.page == "home":
    st.title("Hoş Geldiniz")
    st.write("Lütfen giriş yapın veya üye olun.")
    if st.button("Giriş Yap"):
        go_to_page("login")
    if st.button("Üye Ol"):
        go_to_page("signup")

# --- Login Page ---
elif st.session_state.page == "login":
    st.title("Giriş Yap")
    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Şifre", type="password")
    if st.button("Giriş"):
        if login_user(username, password):
            st.session_state.username = username
            go_to_page("dashboard")
        else:
            st.error("Giriş başarısız. Lütfen kullanıcı adı ve şifrenizi kontrol edin.")

# --- Signup Page ---
elif st.session_state.page == "signup":
    st.title("Üye Ol")
    new_username = st.text_input("Kullanıcı Adı")
    new_password = st.text_input("Şifre", type="password")
    if st.button("Kayıt Ol"):
        add_user(new_username, new_password)
        st.success("Kayıt başarılı. Lütfen giriş yapın.")

# --- Dashboard Page ---
elif st.session_state.page == "dashboard":
    st.title("Dashboard")
    st.write(f"Hoş geldiniz, {st.session_state.get('username','Kullanıcı')}!")
    st.write("Datasetinizi yükleyin, analiz edin ve model eğitin.")

    uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        df = pd.read_csv(st.session_state.uploaded_file)
        st.write("Yüklenen veri:")
        st.dataframe(df.head())

        # --- Analiz Türleri ---
        analysis_type = st.selectbox("Analiz Türü Seçin", 
                                     ["Veri Özeti", "Histogram", "Korelasyon Isı Haritası", "Ortalama Hesaplama"])
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if analysis_type == "Veri Özeti":
            st.subheader("Veri Özeti")
            st.write(df.describe())
        elif analysis_type == "Histogram":
            st.subheader("Histogram")
            if len(numeric_columns) == 0:
                st.warning("Sayısal sütun bulunamadı.")
            else:
                column = st.selectbox("Histogram için sayısal sütun seçin", numeric_columns)
                if column:
                    fig, ax = plt.subplots()
                    sns.histplot(df[column].dropna(), kde=True, ax=ax)
                    st.pyplot(fig)
        elif analysis_type == "Korelasyon Isı Haritası":
            st.subheader("Korelasyon Isı Haritası")
            if len(numeric_columns) == 0:
                st.warning("Sayısal sütun bulunamadı.")
            else:
                fig, ax = plt.subplots()
                sns.heatmap(df[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        elif analysis_type == "Ortalama Hesaplama":
            st.subheader("Ortalama Hesaplama")
            if len(numeric_columns) == 0:
                st.warning("Sayısal sütun bulunamadı.")
            else:
                selected_column = st.selectbox("Ortalama alınacak sütunu seçin", numeric_columns)
                if selected_column:
                    st.write(f"{selected_column} sütununun ortalaması: {df[selected_column].mean():.2f}")

        # --- Model Eğitimi ---
        target_column = st.selectbox("Hedef sütunu seçin", df.columns)
        if st.button("Modeli Eğit"):
            x = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
            y = df[target_column]
            
            if x.empty:
                st.error("Model eğitimi için sayısal sütun bulunamadı.")
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor()
                model.fit(x_train, y_train)
                
                joblib.dump(model, "model.pkl")
                joblib.dump(x.columns.tolist(), "features.pkl")
                st.success("Model başarıyla eğitildi.")
                
                y_pred = model.predict(x_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"R2 Score: {r2:.2f}")
                
                go_to_page("prediction")

# --- Prediction Page ---
elif st.session_state.page == "prediction":
    st.title("Tahmin Yap")

    if not os.path.exists("model.pkl") or not os.path.exists("features.pkl"):
        st.warning("Model bulunamadı. Lütfen önce modeli eğitin.")
        st.stop()

    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")

    # --- Yeni CSV yükleme sadece tahmin için ---
    uploaded_pred_file = st.file_uploader("Tahmin için CSV yükleyin", type=["csv"], key="pred_file")
    
    if uploaded_pred_file is not None:
        pred_df = pd.read_csv(uploaded_pred_file)
        st.write("Yüklenen veri:")
        st.dataframe(pred_df.head())
        
        for col in features:
            if col not in pred_df.columns:
                pred_df[col] = 0
        
        pred_df = pred_df[features]
        pred_df = pred_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        
        predictions = model.predict(pred_df)
        pred_df["Tahmin"] = predictions
        st.subheader("Tahmin Sonuçları")
        st.dataframe(pred_df)
    else:
        st.info("Tahmin yapmak için lütfen bir CSV dosyası yükleyin.")


