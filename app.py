import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Know Your Case - Legal Chatbot", layout="wide")
st.title("Know Your Case: Chat with Legal Documents")

st.markdown("""
Upload your legal documents (e.g., case files, contracts, judgments), and ask questions.
This app uses advanced AI to help you understand your legal documents better.
""")

# Upload documents
uploaded_files = st.file_uploader("Upload legal documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        suffix = file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if suffix == "pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix == "docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == "txt":
            loader = TextLoader(tmp_path)
        else:
            st.warning(f"Unsupported file format: {suffix}")
            continue

        documents = loader.load()
        all_docs.extend(documents)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Use local Hugging Face model via transformers pipeline
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1  # CPU only
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # RAG chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    st.success("✅ Documents processed successfully. You can now chat below.")

    query = st.text_input("Ask a question about your legal document:")

    if query:
        with st.spinner("Searching and generating response..."):
            result = qa_chain({"query": query})
            st.markdown(f"**Answer:** {result['result']}")

            with st.expander("🔍 Sources"):
                for doc in result['source_documents']:
                    st.write(f"📄 **Source**: {doc.metadata.get('source', 'Uploaded File')}")
                    st.write(doc.page_content[:300] + "...")
else:
    st.info("Upload at least one document to begin.")
