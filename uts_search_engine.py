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
st.set_page_config(page_title="Mini Search Engine UTS", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700;800&display=swap');

        /* App Background Override for Humanistic feel */
        .stApp {
            background-color: #FAFAF8; /* Soft warm ivory background */
            background-image: radial-gradient(circle at 50% 0%, #FFF5EB 0%, transparent 70%);
        }
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #1E1A17; /* Soft warm dark brown */
                background-image: radial-gradient(circle at 50% 0%, #2C2621 0%, transparent 70%);
            }
        }

        /* Layout & Spacing */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 3rem !important;
            max-width: 900px;
        }
        
        /* Modern Humanistic Typography */
        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif !important;
            color: #4A433B; /* Warm dark grey */
        }

        /* Hero Header */
        .hero-container {
            text-align: center;
            margin-bottom: 3.5rem;
            animation: fadeInDown 0.8s ease-out;
        }

        .hero-title {
            font-size: 3.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #E27D60 0%, #E8A87C 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            line-height: 1.2;
            letter-spacing: -0.5px;
        }

        .hero-subtitle {
            color: #8C8279;
            font-size: 1.25rem;
            font-weight: 500;
        }

        /* Result Card - Soft & Organic */
        .result-card {
            padding: 24px 28px;
            margin-bottom: 24px;
            border-radius: 20px;
            background: rgba(255, 252, 248, 0.9);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(215, 204, 192, 0.4);
            box-shadow: 0 10px 30px -10px rgba(140, 130, 121, 0.15);
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            animation: fadeInUp 0.6s ease-out backwards;
        }

        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px -10px rgba(226, 125, 96, 0.2);
            border-color: rgba(226, 125, 96, 0.3);
        }
        
        .result-meta {
            font-size: 0.9rem;
            color: #A39A92;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 600;
        }
        
        .result-title {
            font-size: 1.45rem;
            color: #382E26;
            text-decoration: none;
            margin-bottom: 10px;
            font-weight: 700;
            display: inline-block;
            transition: color 0.3s ease;
        }

        .result-title:hover {
            color: #E27D60;
        }
        
        .result-snippet {
            font-size: 1.05rem;
            color: #5D534A;
            line-height: 1.6;
        }
        
        /* Highlight term */
        .highlight {
            background-color: rgba(232, 168, 124, 0.25);
            color: #C0583A;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 6px;
            transition: background-color 0.3s ease;
        }
        .highlight:hover {
            background-color: rgba(232, 168, 124, 0.4);
        }
        
        /* Badge Score */
        .score-badge {
            font-size: 0.8rem;
            background: linear-gradient(135deg, #FDF4E3 0%, #FBE9E7 100%);
            color: #D84315;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            letter-spacing: 0.3px;
            border: 1px solid rgba(216, 67, 21, 0.1);
        }
        
        /* Search metrics info */
        .search-metrics {
            color: #A39A92;
            font-size: 1rem;
            margin-bottom: 24px;
            font-weight: 500;
            animation: fadeIn 0.6s ease-out;
            text-align: center;
        }

        /* Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(25px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-25px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Streamlit Input Enhancements */
        div[data-baseweb="input"] {
            border-radius: 16px;
            background-color: rgba(255, 252, 248, 0.9) !important;
            border: 2px solid rgba(215, 204, 192, 0.5);
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(140, 130, 121, 0.05);
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #E27D60;
            box-shadow: 0 0 0 4px rgba(226, 125, 96, 0.15);
        }

        /* Button Enhancements */
        button[kind="primary"] {
            background: linear-gradient(135deg, #E27D60 0%, #C0583A 100%) !important;
            border: none !important;
            border-radius: 16px !important;
            font-weight: 700 !important;
            color: white !important;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            box-shadow: 0 6px 15px rgba(226, 125, 96, 0.25) !important;
        }
        button[kind="primary"]:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 20px rgba(226, 125, 96, 0.35) !important;
        }
        button[kind="primary"]:active {
            transform: translateY(0) !important;
        }
        
        /* Hiding Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Dark mode compatibility - Warm Dark */
        @media (prefers-color-scheme: dark) {
            html, body, [class*="css"] {
                color: #D8CCC0 !important;
            }
            .result-card {
                background: rgba(44, 38, 33, 0.85);
                border-color: rgba(216, 204, 192, 0.1);
                box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
            }
            .hero-title {
                background: linear-gradient(135deg, #F2A68D 0%, #E8A87C 100%);
                -webkit-background-clip: text;
            }
            .hero-subtitle { color: #A39A92; }
            .result-title { color: #F5EFEB; }
            .result-title:hover { color: #F2A68D; }
            .result-snippet { color: #C5BCB3; }
            .result-meta { color: #8C8279; }
            .highlight {
                background-color: rgba(242, 166, 141, 0.2);
                color: #F2A68D;
            }
            .highlight:hover {
                background-color: rgba(242, 166, 141, 0.3);
            }
            .score-badge { 
                background: linear-gradient(135deg, #4A3A34 0%, #5D433A 100%);
                color: #F2A68D;
                border: 1px solid rgba(242, 166, 141, 0.2);
                box-shadow: none;
            }
            div[data-baseweb="input"] {
                background-color: rgba(33, 28, 24, 0.9) !important;
                border-color: rgba(216, 204, 192, 0.1);
            }
        }
    </style>
    """, unsafe_allow_html=True)

# Panggil fungsi injeksi CSS
inject_custom_css()

st.markdown('''
<div class="hero-container">
    <div class="hero-title">Mini Search Engine</div>
    <div class="hero-subtitle">Cerdas, Cepat, dan Akurat untuk Penelusuran Dokumen TKI</div>
</div>
''', unsafe_allow_html=True)

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
            highlighted.append(f"<span class='highlight'>{w}</span>")
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
    # Modern Metric Card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Dokumen Terindeks", value=len(documents), delta="Ready to search", delta_color="normal")
    
    # Sidebar Setup
    st.sidebar.markdown("### Tentang Aplikasi")
    with st.sidebar.expander("Detail Algoritma", expanded=True):
        st.markdown(r"""
        Diimplementasikan dari awal (*from scratch*):
        - **Pre-processing**: Case folding, Punctuation removal, Stop-word removal, Sastrawi Stemming.
        - **Indexing**: Custom Inverted Index.
        - **Pembobotan**: TF-IDF dengan *Log Frequency* $1 + \log_{10}(tf)$.
        - **Retrieval**: VSM dengan *Cosine Similarity*.
        """)
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pengaturan")
    # Menghapus slider Top-K, karena akan menggunakan Load More
    st.sidebar.info("Aplikasi kini menggunakan sistem 'Load More' untuk memuat hasil pencarian.")

    # UI Kueri Pencarian (Side-by-side)
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_btn = st.columns([4, 1])
    
    with col_input:
        query = st.text_input("Pencarian", label_visibility="collapsed", placeholder="Ketik kata kunci di sini...")
        
    with col_btn:
        search_clicked = st.button("Penelusuran", use_container_width=True, type="primary")
    
    # Divider untuk memisahkan area pencarian dengan hasil
    st.markdown("<hr style='margin-top: 1rem; margin-bottom: 1rem; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Inisialisasi Session State
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'query_tokens' not in st.session_state:
        st.session_state.query_tokens = []
    if 'display_count' not in st.session_state:
        st.session_state.display_count = 10
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'search_time' not in st.session_state:
        st.session_state.search_time = 0

    # Kondisi memicu pencarian baru: tombol diklik ATAU isi query berubah dari query terakhir
    trigger_search = search_clicked or (query.strip() != "" and query != st.session_state.last_query)

    if trigger_search:
        if query.strip():
            start_time = time.time()
            with st.spinner("Mencari dokumen yang relevan..."):
                results, q_tokens = search(query, tfidf_index, idf_dict, doc_vectors, doc_lengths, documents)
            end_time = time.time()
            
            # Simpan hasil ke session_state
            st.session_state.search_results = results
            st.session_state.query_tokens = q_tokens
            st.session_state.display_count = 10
            st.session_state.last_query = query
            st.session_state.search_time = end_time - start_time
        else:
            st.warning("Silakan masukkan kata kunci pencarian terlebih dahulu.")
            st.session_state.search_results = None

    # Proses menampilkan hasil jika ada di session_state
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        q_tokens = st.session_state.query_tokens
        search_time = st.session_state.search_time
        
        # Cek Out-of-Vocabulary (OOV)
        oov_tokens = [t for t in q_tokens if t not in idf_dict]
        if oov_tokens:
            st.warning(f"Kata kunci berikut tidak ditemukan dalam korpus: **{', '.join(oov_tokens)}**")
            
        if results:
            # Menampilkan total keseluruhan metrics
            st.markdown(f"<div class='search-metrics'>Sekitar {len(results)} keseluruhan hasil ditemukan ({search_time:.3f} detik)</div>", unsafe_allow_html=True)
            
            # Membatasi jumlah yang dirender sesuai display_count saat ini
            displayed_results = results[:st.session_state.display_count]
            
            # Render hasil dalam bentuk kartu modern
            for i, res in enumerate(displayed_results):
                highlighted_text = highlight_keywords(res['content'], q_tokens)
                
                card_html = f"""
                <div class="result-card" style="animation-delay: {i * 0.05}s">
                    <div class="result-meta">
                        <span>Peringkat {i+1}</span>
                        <span>•</span>
                        <span>Doc ID: {res['doc_id']}</span>
                        <span class="score-badge">Skor: {res['score']:.4f}</span>
                    </div>
                    <div class="result-title">Dokumen {res['doc_id']}</div>
                    <div class="result-snippet">{highlighted_text}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
            # Tombol "Load More"
            if st.session_state.display_count < len(results):
                st.markdown("<br>", unsafe_allow_html=True)
                col_btn_load_empty1, col_btn_load, col_btn_load_empty2 = st.columns([1, 2, 1])
                with col_btn_load:
                    if st.button("Memuat Lebih Banyak (Load More)", use_container_width=True):
                        st.session_state.display_count += 10
                        if hasattr(st, 'rerun'):
                            st.rerun()
                        else:
                            st.experimental_rerun()
        else:
            st.info("Tidak ditemukan dokumen yang cocok dengan kata kunci pencarian Anda.")
else:
    st.info("Pastikan file 'TKI Sustainability Ecosystem.xlsx' ada di dalam folder yang sama dengan skrip ini.")

