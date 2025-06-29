
import os
import pandas as pd
import time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai

# Ortam Değişkenlerini Yükle
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Pinecone Ayarları
INDEX_NAME = "my-index"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud="aws", region="us-east-1")

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=768, metric="dotproduct", spec=spec)

index = pc.Index(INDEX_NAME)
time.sleep(1)

# Veri ve Embedding
data = pd.read_csv("data/cleaned_products_catalog.csv")
texts = data["Description"].tolist()
metadatas = data[["ProductID", "ProductName"]].to_dict(orient="records")

embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = embedder.embed_documents(texts)

vectors = [
    (str(metadatas[i]["ProductID"]), embeddings[i], metadatas[i])
    for i in range(len(embeddings))
]
index.upsert(vectors=vectors)

# Vector Store
vectorstore = PineconeVectorStore(index=index, embedding=embedder, text_key="Description")

# Streamlit Başlat
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🧠 RAG Chatbot with Gemini")

if "history" not in st.session_state:
    st.session_state.history = []

def get_context(query):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

def generate_answer(question):
    context = get_context(question)
    prompt = f"""
    Context:
    {context}

    Soru: {question}
    Cevap:
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

query = st.text_input("Soru yaz:")

if st.button("Gönder") and query:
    yanit = generate_answer(query)
    st.session_state.history.append(("Sen", query))
    st.session_state.history.append(("Bot", yanit))

for kisi, mesaj in st.session_state.history:
    st.write(f"**{kisi}:** {mesaj}")
