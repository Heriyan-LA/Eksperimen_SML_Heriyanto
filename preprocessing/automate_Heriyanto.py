import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(filepath):
    """Membaca dataset movie rekomendasi"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data berhasil dimuat: {filepath} ({df.shape[0]} baris)")
        return df
    except Exception as e:
        raise FileNotFoundError(f"❌ Gagal memuat data: {e}")

def clean_text(text):
    """Membersihkan teks dari karakter khusus dan lowercase"""
    if pd.isnull(text):
        return ''
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def prepare_content(df):
    """Menggabungkan dan membersihkan teks dari beberapa kolom untuk content-based filtering"""
    required_cols = ['title', 'genres']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Kolom '{col}' tidak ditemukan di dataset.")
    
    # Gabungkan teks dari kolom yang relevan
    df['content'] = df['title'].fillna('') + ' ' + df['genres'].fillna('')
    df['content'] = df['content'].apply(clean_text)
    return df

def vectorize_content(df, max_features=5000):
    """Mengubah teks content menjadi matriks TF-IDF"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df['content'])
    print(f"🔠 TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, tfidf

def reduce_dimensions(tfidf_matrix, n_components=100):
    """Melakukan reduksi dimensi dengan TruncatedSVD"""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    print(f"📉 Dimensi setelah SVD: {reduced_matrix.shape}")
    return reduced_matrix, svd

def compute_similarity(reduced_matrix):
    """Menghitung cosine similarity antar film"""
    sim_matrix = cosine_similarity(reduced_matrix)
    print(f"🤝 Matriks cosine similarity: {sim_matrix.shape}")
    return sim_matrix

def preprocess_pipeline(filepath):
    """
    Pipeline preprocessing otomatis:
    - Load dataset
    - Persiapkan content
    - TF-IDF
    - SVD
    - Cosine Similarity
    """
    df = load_dataset(filepath)
    df = prepare_content(df)
    tfidf_matrix, tfidf_vectorizer = vectorize_content(df)
    reduced_matrix, svd_model = reduce_dimensions(tfidf_matrix)
    similarity_matrix = compute_similarity(reduced_matrix)

    print("✅ Preprocessing selesai. Data siap digunakan untuk rekomendasi.")
    return df, tfidf_matrix, reduced_matrix, similarity_matrix, tfidf_vectorizer, svd_model

# Contoh penggunaan langsung
if __name__ == "__main__":
    filepath = "movie_rekomendasi.csv"
    df, tfidf_matrix, reduced_matrix, similarity_matrix, tfidf, svd = preprocess_pipeline(filepath)
