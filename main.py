import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import os

# Path file langsung
file_path = r"E:\Penting Banget\Semester 5\reksis\Reksis\app\scraping.csv"

# Fungsi untuk pembersihan teks
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    """ Membersihkan teks dengan langkah-langkah pembersihan """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = clean_spcl.sub(' ', text)  # Menghapus karakter spesial
    text = clean_symbol.sub('', text)  # Menghapus simbol selain angka dan huruf
    text = stemmer.stem(text)  # Melakukan stemming
    text = ' '.join(word for word in text.split() if word not in sastrawi.get_stop_words())  # Menghapus stopword
    return text

# *Cek file tersedia*
if os.path.exists(file_path):
    try:
        # Membaca dataset
        laptop_df = pd.read_csv(file_path)
        st.success("File berhasil dimuat!")

        # Validasi kolom 'Product Name'
        if 'product_name' in laptop_df.columns:
            laptop_df['desc_clean'] = laptop_df['product_name'].apply(clean_text)

            # Menghitung TF-IDF dan Cosine Similarity
            laptop_df.set_index('product_name', inplace=True)
            tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
            tfidf_matrix = tf.fit_transform(laptop_df['desc_clean'])
            cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            indices = pd.Series(laptop_df.index)

            # Fungsi rekomendasi produk
            def recomendation(keyword):
                recommended_items = []
                if not indices[indices.str.contains(keyword, case=False, na=False)].empty:
                    matching_products = indices[indices.str.contains(keyword, case=False, na=False)]
                    base_product = matching_products.iloc[0]
                    idx = indices[indices == base_product].index[0]
                    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
                    top_indexes = list(score_series.iloc[1:].index)

                    for i in top_indexes:
                        product_name = laptop_df.index[i]
                        similarity_score = score_series[i]
                        result = f"{product_name} - {similarity_score:.2f}"
                        if result not in recommended_items:
                            recommended_items.append(result)

                    return recommended_items
                else:
                    return f"Tidak ada produk yang cocok dengan kata kunci '{keyword}'."

            # Layout aplikasi Streamlit
            st.title("Sistem Rekomendasi Produk")
            st.sidebar.header("Opsi Pencarian")

            # Input kata kunci
            keyword = st.sidebar.text_input("Masukkan kata kunci untuk pencarian:")

            if keyword:
                recommendations = recomendation(keyword)
                if isinstance(recommendations, list):
                    st.write(f"Rekomendasi untuk '{keyword}':")
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write(recommendations)
        else:
            st.error("Kolom 'product_name' tidak ditemukan dalam dataset.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.error(f"File tidak ditemukan di path: {file_path}")