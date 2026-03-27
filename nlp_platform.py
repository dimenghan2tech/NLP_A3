import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# --- 环境兼容性补丁 ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 资源预加载 (带缓存) ---
@st.cache_resource
def load_nlp_resources():
    # 1. 下载 NLTK 数据
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)
    
    # 2. 预加载小型 GloVe 模型 (Twitter 25d 仅 30MB 左右，适合部署)
    try:
        glove = api.load("glove-twitter-25")
    except Exception as e:
        glove = None
        st.error(f"GloVe 加载失败: {e}")
    return glove

glove_model = load_nlp_resources()

# --- 页面 UI 配置 ---
st.set_page_config(page_title="NLP 语义分析平台", layout="wide")
st.title("🚀 NLP 语义分析综合测试平台")
st.info("提示：如果首次运行加载较慢，请稍等片刻，系统正在初始化模型资源。")

# 侧边栏：语料输入
st.sidebar.header("输入语料库")
default_text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence... (此处省略你的长文本)"""
corpus_text = st.sidebar.text_area("输入英文文本:", value=default_text, height=300)

tab1, tab2, tab3, tab4 = st.tabs(["TF-IDF & LSA", "Word2Vec", "GloVe 类比", "FastText 词向量"])

# --- 模块 1: TF-IDF & LSA ---
with tab1:
    st.header("1. 传统统计模型")
    if corpus_text:
        sentences = sent_tokenize(corpus_text)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("TF-IDF 关键词")
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            top_indices = avg_tfidf.argsort()[::-1][:5]
            st.table(pd.DataFrame({
                "关键词": [vectorizer.get_feature_names_out()[i] for i in top_indices],
                "得分": [avg_tfidf[i] for i in top_indices]
            }))
        
        with col2:
            st.subheader("LSA 2D 语义空间可视化")
            lsa = TruncatedSVD(n_components=2)
            lsa_matrix = lsa.fit_transform(tfidf_matrix.T)
            fig, ax = plt.subplots()
            plot_indices = avg_tfidf.argsort()[::-1][:20]
            for i in plot_indices:
                ax.scatter(lsa_matrix[i, 0], lsa_matrix[i, 1])
                ax.text(lsa_matrix[i, 0], lsa_matrix[i, 1], vectorizer.get_feature_names_out()[i])
            st.pyplot(fig)

# --- 模块 2: Word2Vec ---
with tab2:
    st.header("2. Word2Vec 训练")
    if corpus_text:
        tokens = [word_tokenize(s.lower()) for s in sent_tokenize(corpus_text)]
        arch = st.radio("模型架构", ["CBOW", "Skip-Gram"], horizontal=True)
        sg_val = 1 if arch == "Skip-Gram" else 0
        
        model_w2v = Word2Vec(sentences=tokens, vector_size=100, sg=sg_val, min_count=1)
        word = st.text_input("查找相似词:", value="language")
        if word.lower() in model_w2v.wv:
            sims = model_w2v.wv.most_similar(word.lower(), topn=5)
            st.table(pd.DataFrame(sims, columns=["单词", "相似度"]))

# --- 模块 3: GloVe ---
with tab3:
    st.header("3. GloVe 词汇类比")
    if glove_model:
        st.write("公式: A - B + C = ? (例如: king - man + woman = queen)")
        c1, c2, c3 = st.columns(3)
        a = c1.text_input("A", "king")
        b = c2.text_input("B", "man")
        c = c3.text_input("C", "woman")
        if st.button("计算类比"):
            try:
                res = glove_model.most_similar(positive=[a, c], negative=[b], topn=1)
                st.success(f"预测结果: **{res[0][0]}**")
            except Exception as e:
                st.error(f"词汇不在字典中: {e}")
    else:
        st.warning("GloVe 模型未加载。")

# --- 模块 4: FastText ---
with tab4:
    st.header("4. FastText (处理 OOV 词汇)")
    if corpus_text:
        tokens = [word_tokenize(s.lower()) for s in sent_tokenize(corpus_text)]
        model_ft = FastText(sentences=tokens, vector_size=100, min_count=1)
        
        oov = st.text_input("输入一个拼写错误的词 (如 computeer):", "computeer")
        if st.button("测试 FastText 联想"):
            res = model_ft.wv.most_similar(oov.lower(), topn=3)
            st.table(pd.DataFrame(res, columns=["候选词", "相似度"]))
