import streamlit as st
import os
import cv2
import uuid
from PIL import Image
from traditional_cv import detect_copy_move_forgery
from ai_models import analyze_with_ai

# Klasör yollarını Windows'ta kesin çalışacak şekilde mutlak yola (Absolute Path) çeviriyoruz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, "data", "original_images")
os.makedirs(TEMP_DIR, exist_ok=True)

st.set_page_config(page_title="Görüntü Sahteciliği Tespiti", page_icon="🔍", layout="centered")

st.title("🔍 Görüntü Sahteciliği Tespiti")
st.markdown("Bu araç, resimler üzerindeki sahtecilikleri geleneksel bilgisayarlı görü ve yapay zeka (AI) algoritmaları ile tespit etmek için tasarlanmıştır.")

uploaded_file = st.file_uploader("Lütfen analiz edilecek görseli yükleyin...", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    st.subheader("Yüklenen Görsel")
    image = Image.open(uploaded_file)
    st.image(image, caption="Orijinal Görsel", use_container_width=True)

    
    file_extension = os.path.splitext(uploaded_file.name)[1]
    safe_filename = str(uuid.uuid4()) + file_extension
    temp_file_path = os.path.join(TEMP_DIR, safe_filename)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.divider()
    st.subheader("🛠️ Analiz Seçenekleri")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        algorithm = st.selectbox(
            "Kullanılacak Algoritmayı Seçin:",
            ("SIFT", "AKAZE", "ORB", "CNN", "LSTM (CNN-LSTM)") 
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("Görüntüyü Analiz Et", use_container_width=True)

    if analyze_button:
        with st.spinner(f"{algorithm} algoritması ile analiz ediliyor..."):
            
            # 1. Geleneksel Algoritmalar
            if algorithm in ["SIFT", "AKAZE", "ORB"]:
                result_img = detect_copy_move_forgery(temp_file_path, algorithm=algorithm)
                if result_img is not None:
                    st.success("Analiz başarıyla tamamlandı!")
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.subheader("Analiz Sonucu")
                    st.image(result_rgb, caption=f"{algorithm} Algoritması Sonucu", use_container_width=True)
                else:
                    st.warning("Görsel üzerinde sahteciliğe dair yeterli eşleşme bulunamadı veya resim algoritmaya uygun değil.")
            
            # 2. Yapay Zeka Algoritmaları
            elif algorithm in ["CNN", "LSTM (CNN-LSTM)"]:
                is_forged, confidence = analyze_with_ai(temp_file_path, model_type=algorithm)
                
                # Okuma hatasını yakalama kontrolü eklendi
                if is_forged is None:
                    st.error("🚨 HATA: Görsel arka planda okunamadı! Lütfen farklı bir görsel deneyin.")
                else:
                    st.success("Yapay Zeka Analizi Tamamlandı!")
                    if is_forged:
                        st.error(f"🚨 **DİKKAT:** Bu görselin sahte (manipüle edilmiş) olma ihtimali: **%{confidence:.2f}**")
                    else:
                        st.info(f"✅ Bu görselin orijinal olma ihtimali: **%{confidence:.2f}**")