import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2') 

st.title("📄 Ask My PDF Bot")
st.write("Sreamlit is working successfully")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    chunks = text.split("\n")

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))
if uploaded_file is not None:

    pdf_reader = PdfReader(uploaded_file)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    chunks = text.split(". ")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    question = st.text_input("Ask a question from PDF")

    if question:
        q_embedding = model.encode([question])

        _, idx = index.search(
            np.array(q_embedding).astype('float32'),
            3
        )

        context = ""

        for i in idx[0]:
            context += chunks[i] + " "

        st.write("Answer:")
        st.write(context)
    