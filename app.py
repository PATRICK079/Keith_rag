import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Streamlit UI
st.set_page_config(page_title="Complaint Analyzer with Gemini", layout="wide")
st.title("📣 Customer Complaint Analyzer")
st.markdown("Ask anything about customer complaints stored in this CSV.")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("complaints.csv")
    df = df.fillna("")
    df["text"] = df.apply(
        lambda row: f"The customer filed a complaint about {row['product']}. Here's what they said: {row['narrative']}",
        axis=1
    )
    return df

df = load_data()
st.write("### Preview of complaints data", df.head())

# Prepare embeddings
@st.cache_resource
def get_vectorstore():
    # Split text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunked_docs = []
    for text in df["text"].tolist():
        chunks = text_splitter.split_text(text)
        chunked_docs.extend([Document(page_content=chunk) for chunk in chunks])

    # Embed and store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunked_docs, embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

vectordb = get_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Prompt template
template = """
You are a helpful AI assistant.
Use the following complaint records to answer the question.
If the answer is not found, say "I couldn't find that in the complaints."

Context:
{context}

Question: {input}
Answer:
"""
prompt = PromptTemplate.from_template(template)

# Gemini Model (use your API key)
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, convert_system_message_to_human=True)

# Build Retrieval Chain
combine_docs_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# User input
query = st.text_input("🔍 Ask a question about the complaints:")
if query:
    with st.spinner("Thinking..."):
        response = retrieval_chain.invoke({"input": query})
        st.markdown("### 💬 Answer")
        st.write(response["answer"])
