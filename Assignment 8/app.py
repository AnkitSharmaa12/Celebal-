
import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="RAG Q&A Chatbot", layout="wide")

#  custom CSS 
page_bg_color = '''
<style>
    /* Entire app background */
    .stApp {
        background-color: #f0f8ff;
        background-image: none;
        background-size: cover;
    }

    /* Optional: style the text input box */
    input[type="text"] {
        background-color: #ffffff;
        color: #000000;
    }

    /* Optional: style sidebar (if used) */
    .css-1d391kg {
        background-color: #ffffff !important;
    }
</style>
'''
st.markdown(page_bg_color, unsafe_allow_html=True)

# Load embedding model

@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        return None

# Load FAISS index and documents
@st.cache_resource
def load_embeddings():
    try:
        with open("documents.pkl", "rb") as f:
            documents = pickle.load(f)
        with open("faiss_index.pkl", "rb") as f:
            index = pickle.load(f)
        return documents, index
    except Exception as e:
        st.error(f"❌ Error loading embeddings or index: {e}")
        return None, None

# Perform semantic search
def search(query, model, index, documents, top_k=3):
    if model is None or index is None or documents is None:
        return ["❌ Model or data not loaded properly."]
    try:
        query_embedding = model.encode([query])
        scores, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
        results = [documents[i] for i in indices[0]]
        return results
    except Exception as e:
        return [f"❌ Search error: {e}"]

# Page setup
st.set_page_config(page_title="Loan Q&A Chatbot", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .answer-box {
            background-color: #ffffff;
            border-left: 5px solid #00c0a0;
            padding: 10px 15px;
            margin-bottom: 12px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .faq-box {
            background-color: #f9f9f9;
            padding: 10px 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #8884d8;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Loan Data Q&A Chatbot (RAG + HuggingFace)")

# Load data
model = load_model()
documents, index = load_embeddings()

# Create two columns: left (chat), right (FAQ)
left_col, right_col = st.columns([2.5, 1])

# ==== Left Column (Main Chat) ====
with left_col:
    st.markdown("Ask any question about the loan approval dataset ⬇️")
    user_query = st.text_input("💬 Your question:")

    if user_query:
        with st.spinner("🔍 Searching best answers..."):
            responses = search(user_query, model, index, documents)
            st.markdown("---")
            st.subheader("📚 Top Answers")
            for i, res in enumerate(responses):
                    
                    if i == 0:
                        st.markdown(
                            f"<div class='answer-box' style='background-color:#fff9e6; border-left: 5px solid #f7c948;'>"
                            f"<b>{i+1}.</b> {res}</div>", unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='answer-box'><b>{i+1}.</b> {res}</div>",
                            unsafe_allow_html=True
                        )


# ==== Right Column (FAQs) ====
with right_col:
    st.subheader("📌 FAQs")
    with st.expander("View Common Questions"):
        faqs = [
            "What are the factors affecting loan approval?",
            "What percentage of applicants are self-employed?",
            "Is there any gender bias in loan approvals?",
            "Which credit history scores get more approvals?",
            "How does income relate to loan status?",
            "What education level gets more approvals?",
            "Does property area affect approval?",
            "Are married applicants more likely to get loans?",
        ]
        for i, question in enumerate(faqs):
            st.markdown(f"<div class='faq-box'>✅ {question}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center>✨ Built with ❤️ using HuggingFace, FAISS & Streamlit</center>", unsafe_allow_html=True)
