import streamlit as st
import pandas as pd
import re
import math
import time
from collections import defaultdict, Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- SETUP PAGE ---
st.set_page_config(page_title="Mini Search Engine UTS", page_icon="🔍", layout="wide")

st.title("🔍 Mini Search Engine (UTS TKI)")
st.markdown(r"""
Aplikasi ini diimplementasikan dari awal (*from scratch*) sesuai dengan kebutuhan UTS Temu Kembali Informasi:
- **Pre-processing**: Case folding, Punctuation removal, NLTK Stop-word removal, Sastrawi Stemming.
- **Indexing**: Custom Inverted Index.
- **Pembobotan**: TF-IDF dengan *Log Frequency Weighting* $1 + \log_{10}(tf)$.
- **Retrieval**: Vector Space Model dengan *Cosine Similarity*.
""")

# --- NLTK & SASTRAWI INITIALIZATION ---
@st.cache_resource
def load_nlp_tools():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stop_words, stemmer

stop_words, stemmer = load_nlp_tools()

# --- 1. PRE-PROCESSING ---
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    
    # 1. Case Folding
    text = text.lower()
    
    # 2. Punctuation Removal (hanya ambil alfabet dan angka)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Tokenisasi manual sederhana (split by space)
    tokens = text.split()
    
    # 3. Stop-word Removal & 4. Stemming
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words:
            stemmed_token = stemmer.stem(token)
            if stemmed_token:  # pastikan tidak kosong
                cleaned_tokens.append(stemmed_token)
                
    return cleaned_tokens

# --- 2. INVERTED INDEX ---
def build_inverted_index(documents):
    # document format: {doc_id: text}
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = {} # untuk VSM length normalization
    
    # Pre-process & hitung kemunculan kata (TF mentah)
    processed_docs = {}
    for doc_id, text in documents.items():
        tokens = preprocess_text(text)
        processed_docs[doc_id] = tokens
        
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            inverted_index[term][doc_id] = count
            
    # Convert defaultdict to dict to prevent pickling errors in Streamlit Cache
    inverted_index = {k: dict(v) for k, v in inverted_index.items()}
    return inverted_index, processed_docs

# --- 3. TF-IDF (LOG FREQUENCY WEIGHTING) ---
def compute_tf_idf(inverted_index, num_docs):
    # Struktur: {term: {doc_id: tf_idf_weight}}
    tfidf_index = defaultdict(dict)
    idf_dict = {}
    doc_vectors = defaultdict(dict)
    
    for term, doc_freqs in inverted_index.items():
        # df = jumlah dokumen yang mengandung term
        df = len(doc_freqs)
        # IDF (standar)
        idf = math.log10(num_docs / df) if df > 0 else 0
        idf_dict[term] = idf
        
        for doc_id, tf_raw in doc_freqs.items():
            # Log frequency weighting untuk TF
            tf_weight = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
            
            # W = TF * IDF
            weight = tf_weight * idf
            tfidf_index[term][doc_id] = weight
            doc_vectors[doc_id][term] = weight
            
    # Hitung panjang vektor setiap dokumen (denominator Cosine Similarity)
    doc_lengths = {}
    for doc_id, vector in doc_vectors.items():
        length = math.sqrt(sum(w**2 for w in vector.values()))
        doc_lengths[doc_id] = length
        
    # Convert defaultdict to dict to prevent pickling errors in Streamlit Cache
    tfidf_index = {k: dict(v) for k, v in tfidf_index.items()}
    doc_vectors = {k: dict(v) for k, v in doc_vectors.items()}
        
    return tfidf_index, idf_dict, doc_vectors, doc_lengths

# --- 4. RETRIEVAL (VECTOR SPACE MODEL) ---
def search(query, tfidf_index, idf_dict, doc_vectors, doc_lengths, documents):
    query_tokens = preprocess_text(query)
    
    # Hitung vektor TF-IDF kueri
    query_counts = Counter(query_tokens)
    query_vector = {}
    for term, tf_raw in query_counts.items():
        if term in idf_dict:
            tf_weight = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
            query_vector[term] = tf_weight * idf_dict[term]
            
    # Jika query kosong atau kata tidak ada di index
    if not query_vector:
        return [], query_tokens
        
    # Hitung panjang vektor kueri
    query_length = math.sqrt(sum(w**2 for w in query_vector.values()))
    
    # Hitung Dot Product
    scores = defaultdict(float)
    for term, q_weight in query_vector.items():
        if term in tfidf_index:
            for doc_id, d_weight in tfidf_index[term].items():
                scores[doc_id] += q_weight * d_weight
                
    # Hitung Cosine Similarity
    ranked_results = []
    for doc_id, dot_product in scores.items():
        if doc_lengths[doc_id] > 0 and query_length > 0:
            cosine_sim = dot_product / (query_length * doc_lengths[doc_id])
            if cosine_sim > 0:
                ranked_results.append({
                    "doc_id": doc_id,
                    "score": cosine_sim,
                    "content": documents[doc_id]
                })
                
    # Urutkan berdasarkan score terbesar
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results, query_tokens

def highlight_keywords(text, query_tokens):
    words = text.split()
    highlighted = []
    for w in words:
        clean_w = re.sub(r'[^a-z0-9\s]', '', w.lower())
        stemmed_w = stemmer.stem(clean_w) if clean_w else ""
        if stemmed_w in query_tokens and stemmed_w != "":
            highlighted.append(f"<span style='background-color: yellow; color: black; padding: 0 4px; border-radius: 4px;'><b>{w}</b></span>")
        else:
            highlighted.append(w)
    return " ".join(highlighted)

# --- STREAMLIT APP LOGIC ---
@st.cache_data
def load_and_index_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='IR')
        if 'Data' not in df.columns:
            st.error("Format Excel salah. Pastikan ada kolom bernama 'Data'.")
            return None, None, None, None, None, None
            
        # Bentuk kamus dokumen: {doc_id: text}
        documents = {i: str(text) for i, text in enumerate(df['Data'])}
        num_docs = len(documents)
        
        # Proses IR
        inverted_index, _ = build_inverted_index(documents)
        tfidf_index, idf_dict, doc_vectors, doc_lengths = compute_tf_idf(inverted_index, num_docs)
        
        return documents, inverted_index, tfidf_index, idf_dict, doc_vectors, doc_lengths
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None, None, None, None, None, None

# Load Data
DATA_PATH = 'TKI Sustainability Ecosystem.xlsx'

with st.spinner("Membangun Inverted Index dan menghitung TF-IDF. Harap tunggu..."):
    documents, inverted_index, tfidf_index, idf_dict, doc_vectors, doc_lengths = load_and_index_data(DATA_PATH)

if documents:
    st.success(f"Berhasil mengindeks {len(documents)} dokumen dari {DATA_PATH}.")
    
    # Sidebar
    st.sidebar.markdown("### ⚙️ Pengaturan Pencarian")
    top_k = st.sidebar.slider("Jumlah hasil (Top-K) maksimal:", min_value=1, max_value=50, value=10)

    # UI Kueri Pencarian
    st.markdown("### Masukkan Kata Kunci Pencarian")
    query = st.text_input("Contoh: ekosistem pesisir laut", placeholder="Ketik kata kunci di sini...")
    
    if st.button("Cari", type="primary"):
        if query.strip():
            start_time = time.time()
            with st.spinner("Mencari dokumen yang relevan..."):
                results, q_tokens = search(query, tfidf_index, idf_dict, doc_vectors, doc_lengths, documents)
            end_time = time.time()
            
            # Cek Out-of-Vocabulary (OOV)
            oov_tokens = [t for t in q_tokens if t not in idf_dict]
            if oov_tokens:
                st.warning(f"⚠️ Kata kunci berikut tidak ditemukan dalam korpus dokumen (diabaikan): **{', '.join(oov_tokens)}**")
                
            if results:
                st.success(f"⚡ Ditemukan **{len(results)}** dokumen yang relevan dalam **{end_time - start_time:.3f} detik**.")
                
                # Membatasi hasil Top-K
                results = results[:top_k]
                st.write(f"Menampilkan **Top {len(results)}** dokumen terbaik:")
                
                for i, res in enumerate(results):
                    with st.container(border=True):
                        st.markdown(f"**Peringkat {i+1}** | `Doc ID: {res['doc_id']}` | 🎯 **Skor Kesamaan (Cosine Sim): {res['score']:.4f}**")
                        # Highlighting teks
                        highlighted_text = highlight_keywords(res['content'], q_tokens)
                        st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.error("Tidak ditemukan dokumen yang cocok dengan kata kunci pencarian Anda.")
        else:
            st.warning("Silakan masukkan kata kunci pencarian terlebih dahulu.")
else:
    st.info("Pastikan file 'TKI Sustainability Ecosystem.xlsx' ada di dalam folder yang sama dengan skrip ini.")

