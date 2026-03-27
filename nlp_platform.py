import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')

st.set_page_config(page_title="Semantic Analysis Platform", layout="wide")

st.title("Semantic Analysis Comprehensive Test Platform")
st.markdown("---")

# Sidebar for common input
st.sidebar.header("Input Corpus")
default_text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation. Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems. Word2Vec is a popular method for learning word embeddings from a text corpus. It uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. GloVe is another model for distributed word representation. It is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. FastText is a library for learning of word embeddings and text classification created by Facebook's AI Research lab. It uses a subword model to represent words as bags of character n-grams. This allows the model to handle out-of-vocabulary words by using the representations of their constituent n-grams."""
corpus_text = st.sidebar.text_area("Enter English text (500-1000 words recommended):", value=default_text, height=300)

tab1, tab2, tab3, tab4 = st.tabs(["TF-IDF & LSA", "Word2Vec", "GloVe Analogy", "FastText & Sent2Vec"])

# --- Module 1: TF-IDF & LSA ---
with tab1:
    st.header("Module 1: Traditional Statistical Models")
    if corpus_text:
        sentences = sent_tokenize(corpus_text)
        st.write(f"Total sentences (documents): {len(sentences)}")
        
        # TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Show top keywords
        st.subheader("TF-IDF Top Keywords")
        avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_indices = avg_tfidf.argsort()[::-1][:5]
        top_keywords = pd.DataFrame({
            "Keyword": [feature_names[i] for i in top_indices],
            "Score": [avg_tfidf[i] for i in top_indices]
        })
        st.table(top_keywords)
        
        # LSA
        st.subheader("LSA 2D Visualization")
        lsa = TruncatedSVD(n_components=2)
        # We reduce the word vectors (transpose of TF-IDF matrix)
        lsa_matrix = lsa.fit_transform(tfidf_matrix.T)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot top 30 words for clarity
        plot_indices = avg_tfidf.argsort()[::-1][:30]
        for i in plot_indices:
            ax.scatter(lsa_matrix[i, 0], lsa_matrix[i, 1], alpha=0.5)
            ax.text(lsa_matrix[i, 0], lsa_matrix[i, 1], feature_names[i], fontsize=9)
        
        ax.set_title("LSA Word Projections (Top 30 Words)")
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        st.pyplot(fig)

# --- Module 2: Word2Vec ---
with tab2:
    st.header("Module 2: Word2Vec Training")
    if corpus_text:
        tokenized_sentences = [word_tokenize(s.lower()) for s in sent_tokenize(corpus_text)]
        
        col1, col2 = st.columns(2)
        with col1:
            arch = st.radio("Architecture", ["CBOW", "Skip-Gram"])
            sg_val = 1 if arch == "Skip-Gram" else 0
        with col2:
            win = st.slider("Window Size", 2, 10, 5)
            
        # Train model
        model_w2v = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=win, sg=sg_val, min_count=1, workers=4)
        
        search_word = st.text_input("Find similar words for:", value="language")
        if search_word.lower() in model_w2v.wv:
            similar = model_w2v.wv.most_similar(search_word.lower(), topn=5)
            st.write("Top 5 Similar Words:")
            st.table(pd.DataFrame(similar, columns=["Word", "Similarity"]))
        else:
            st.warning("Word not in vocabulary.")

# --- Module 3: GloVe ---
with tab3:
    st.header("Module 3: GloVe & Word Analogies")
    
    @st.cache_resource
    def load_glove():
        # Using a small model for speed
        return api.load("glove-twitter-25")
    
    try:
        glove_model = load_glove()
        
        st.subheader("Word Analogy (A - B + C = ?)")
        col1, col2, col3 = st.columns(3)
        with col1: a = st.text_input("A (e.g. king)", value="king")
        with col2: b = st.text_input("B (e.g. man)", value="man")
        with col3: c = st.text_input("C (e.g. woman)", value="woman")
        
        if st.button("Compute Analogy"):
            try:
                res = glove_model.most_similar(positive=[a, c], negative=[b], topn=1)
                st.success(f"Result: {res[0][0]} (Score: {res[0][1]:.4f})")
            except KeyError as e:
                st.error(f"Error: {e}")
                
        st.subheader("Similarity Score")
        w1 = st.text_input("Word 1", value="apple")
        w2 = st.text_input("Word 2", value="orange")
        if w1 in glove_model and w2 in glove_model:
            sim = glove_model.similarity(w1, w2)
            st.info(f"Cosine Similarity: {sim:.4f}")
    except Exception as e:
        st.error(f"Failed to load GloVe model: {e}")

# --- Module 4: FastText & Sent2Vec ---
with tab4:
    st.header("Module 4: FastText & Sent2Vec")
    if corpus_text:
        tokenized_sentences = [word_tokenize(s.lower()) for s in sent_tokenize(corpus_text)]
        model_ft = FastText(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        st.subheader("OOV Handling Test")
        oov_word = st.text_input("Enter a word with typo (OOV):", value="computeer")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Word2Vec Result:**")
            try:
                # We use the model from tab2
                vec = model_w2v.wv[oov_word.lower()]
                st.write("Vector found (unexpected).")
            except KeyError:
                st.error("KeyError: Word not in vocabulary")
        
        with col2:
            st.write("**FastText Result:**")
            try:
                sim_ft = model_ft.wv.most_similar(oov_word.lower(), topn=3)
                st.write("Top similar words (via subwords):")
                st.table(pd.DataFrame(sim_ft, columns=["Word", "Similarity"]))
            except Exception as e:
                st.write(f"Error: {e}")
                
        st.subheader("Sent2Vec (Average Pooling)")
        s1 = st.text_area("Sentence 1:", value="Natural language processing is a fascinating field.")
        s2 = st.text_area("Sentence 2:", value="The study of human language with computers is very interesting.")
        
        def get_sent_vec(sentence, model):
            words = word_tokenize(sentence.lower())
            vecs = [model.wv[w] for w in words if w in model.wv]
            if not vecs: return np.zeros(100)
            return np.mean(vecs, axis=0)
            
        if st.button("Calculate Sentence Similarity"):
            v1 = get_sent_vec(s1, model_ft).reshape(1, -1)
            v2 = get_sent_vec(s2, model_ft).reshape(1, -1)
            sim_score = cosine_similarity(v1, v2)[0][0]
            st.success(f"Sentence Similarity: {sim_score:.4f}")
