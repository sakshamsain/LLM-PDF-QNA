import streamlit as st
import os
import cassio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import tempfile

st.set_page_config(page_title="PDF Q&A App", layout="wide")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embedding" not in st.session_state:
    st.session_state.embedding = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    astra_db_token = st.text_input("Astra DB Token", type="password", value="AstraCS:vPwNayHGECkzJdFOoJJSlqBX:b01840b58400c80dba3abcb93a0424f818a1e369472f95cc71974ded64fcbc81")
    astra_db_id = st.text_input("Astra DB ID", value="9a3f2348-92c3-460a-8435-40c3fe8c71c6")
    gemini_api_key = st.text_input("Gemini API Key", type="password", value="AIzaSyDVBi6ylBhZZPcL8FFZ7W2PYe8PQHMA6bQ")

    if st.button("Initialize Database"):
        if astra_db_token and astra_db_id and gemini_api_key:
            try:
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
                cassio.init(token=astra_db_token, database_id=astra_db_id)
                # Initialize LLM and embeddings
                st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
                st.session_state.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.success("Database and models initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing database: {e}")
        else:
            st.error("Please provide all required credentials.")

# Main app
st.title("PDF Question Answering App")
st.write("Upload a PDF file and ask questions based on its content.")

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    if st.session_state.embedding is None:
        st.error("Please initialize the database and models first.")
    else:
        with st.spinner("Processing PDF..."):
            try:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # Read PDF
                pdfreader = PdfReader(temp_file_path)
                raw_text = ""
                for page in pdfreader.pages:
                    content = page.extract_text()
                    if content:
                        raw_text += content

                # Initialize vector store
                st.session_state.vector_store = Cassandra(
                    embedding=st.session_state.embedding,
                    table_name="qa_mini_demo",
                    session=None,
                    keyspace=None
                )

                # Split text
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=800,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(raw_text)

                # Add texts to vector store
                st.session_state.vector_store.add_texts(texts[:50])
                st.session_state.vector_index = VectorStoreIndexWrapper(vectorstore=st.session_state.vector_store)
                st.success(f"Inserted {len(texts[:50])} text chunks into the vector store.")

                # Clean up temporary file
                os.unlink(temp_file_path)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# Question input and answering
if st.session_state.vector_index and st.session_state.llm:
    st.subheader("Ask a Question")
    query_text = st.text_input("Enter your question:")
    if st.button("Get Answer") and query_text:
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.vector_index.query(query_text, llm=st.session_state.llm).strip()
                st.markdown("**Answer:**")
                st.write(answer)

                st.markdown("**Relevant Documents:**")
                docs = st.session_state.vector_store.similarity_search_with_score(query_text, k=4)
                for doc, score in docs:
                    st.write(f"[{score:.4f}] {doc.page_content[:84]}...")
            except Exception as e:
                st.error(f"Error answering question: {e}")
else:
    if uploaded_file is None:
        st.info("Please upload a PDF file to start.")
    elif st.session_state.llm is None or st.session_state.embedding is None:
        st.info("Please initialize the database and models.")
    else:
        st.info("Please process the PDF.")