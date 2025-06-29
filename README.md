# 📦 Tam Entegre RAG Chatbot (Gemini + LangChain + Pinecone + Streamlit)

Bu proje, Google Gemini LLM API'yi kullanarak bir RAG (Retrieval-Augmented Generation) tabanlı chatbot uygulamasıdır. LangChain, Pinecone ve Streamlit teknolojilerini entegre biçimde içerir.

## 📝 README

### 🚀 Proje Özellikleri
- **LangChain** ile vektör tabanlı doküman arama
- **Gemini-Pro** ile doğal dil işleme
- **Pinecone** ile vektör veritabanı
- **Streamlit** ile interaktif web arayüzü

---

### 📁 Dizin Yapısı
```
chatbot-rag-gemini/
├── run.py               → Ana uygulama dosyası
├── requirements.txt    → Gerekli Python kütüphaneleri
├── .env                → API anahtarları (yüklenmez)
├── data/
│   └── cleaned_products_catalog.csv
```

---

### 🔐 .env Dosyası
Proje kök dizinine `.env` dosyasını şu içerikle oluştur:
```
GOOGLE_API_KEY=senin_gemini_keyin
PINECONE_API_KEY=senin_pinecone_keyin
```

> Bu dosya `.gitignore` içine alınmalı ve GitHub'a yüklenmemelidir.

---

### ⚙️ requirements.txt
```
streamlit
langchain
google-generativeai
git+https://github.com/langchain-ai/langchain.git@v0.1.16#subdirectory=libs/community
pinecone
python-dotenv
pandas
```

---

### ⚠️ Önemli Uyarı (Gemini Hatası Hakkında)

Projede sık karşılaşılan bir hata şudur:
```
google.api_core.exceptions.NotFound: 404 models/gemini-pro is not found for API version v1beta
```
Bu hatanın nedeni, `langchain_google_genai` ve `google-generativeai` kullanılırken **model adının yanlış formatta** verilmesidir.

**Yanlış kullanım:**
```python
ChatGoogleGenerativeAI(model="models/gemini-pro")
```

**Doğru kullanım:**
```python
ChatGoogleGenerativeAI(model="gemini-pro")
```
Aynı şekilde embedding kısmında da:
```python
GoogleGenerativeAIEmbeddings(model="embedding-001")
```
şeklinde yazılmalıdır. `models/` öneki **gereksizdir ve hataya neden olur.**

---

### ▶️ Projeyi Çalıştırmak
```bash
pip install -r requirements.txt
streamlit run run.py
```

---

### 📤 GitHub'a Yükleme İpuçları
1. `.gitignore` dosyasına `.env` ve `__pycache__/` gibi klasörleri eklemeyi unutma.
2. README.md ve requirements.txt ile birlikte yükle.
3. Yüklemeden önce `git init`, `git remote add`, `git push` işlemleri ile bağla.

---

### 📌 Ekstra Not
Bu proje demo amaçlıdır. Daha büyük veri kümeleri, cache optimizasyonları, API sınırları ve kimlik doğrulama gibi gelişmiş güvenlik önlemleri için genişletilebilir.

---

## 🔧 Kod Yapısı
```python
# Ortam Değişkenlerini Yükle
load_dotenv()

# Pinecone Ayarları
INDEX_NAME = "my-index"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
...

# Embedding
embedder = GoogleGenerativeAIEmbeddings(model="embedding-001")
...

# LLM modeli
model = ChatGoogleGenerativeAI(model="gemini-pro")
```

Proje, `streamlit` arayüzü üzerinden kullanıcıdan gelen sorulara Gemini destekli cevaplar üretir.

---

### 📞 Yardım veya Geri Bildirim
Herhangi bir hata ya da katkı için lütfen GitHub üzerinden [Issue](https://github.com/) aç.

---

### 📌 Not
Bu README dosyası hem teknik dokümantasyon hem de yükleme rehberi olarak düzenlenmiştir.

---

## Kod (run.py)

```python
import os
import pandas as pd
import time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Ortam Değişkenlerini Yükle
load_dotenv()

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

embedder = GoogleGenerativeAIEmbeddings(model="embedding-001")
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
st.title("🤖 RAG Chatbot (Gemini ile)")

if "history" not in st.session_state:
    st.session_state.history = []

def get_context(query):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

def generate_answer(question):
    context = get_context(question)
    prompt = f"""
    Aşağıdaki bağlam bilgilerine dayanarak kullanıcı sorusunu yanıtla:
    Context:
    {context}

    Soru: {question}
    Cevap:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro")
    response = model.invoke(prompt)
    return response.content.strip()

query = st.text_input("Soru yaz:")

if st.button("Gönder") and query:
    yanit = generate_answer(query)
    st.session_state.history.append(("Sen", query))
    st.session_state.history.append(("Bot", yanit))

for kisi, mesaj in st.session_state.history:
    st.write(f"**{kisi}:** {mesaj}")
```
