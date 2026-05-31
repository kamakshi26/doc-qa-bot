import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import tempfile
import os

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A Bot")
st.caption("Upload any PDF. Ask questions. Get cited answers.")

# ─── Session state ────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ─── PDF Upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
    help="Upload any text-based PDF"
)

@st.cache_resource(show_spinner="Processing PDF — this takes ~30 seconds...")
def build_chain(file_bytes, filename):
    # save to temp file — PyPDFLoader needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp_path = f.name

    # load
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # embed + store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",  # in-memory — no Docker needed
        collection_name="uploaded_doc"
    )

    # prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Answer using ONLY the context below.

        If the answer is not in the context, respond with exactly this and nothing else:
        "I don't know based on the uploaded document."

        If you do find the answer in the context, end your response with:
        Source: [quote the exact sentence from the context that contains the answer]

        Context: {context}

        Question: {question}
        Answer:"""
    )

    # chain
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=st.secrets["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com",
        temperature=0
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    os.unlink(tmp_path)
    return chain, len(chunks)

# ─── Process upload ───────────────────────────────────────────
if uploaded_file:
    if uploaded_file.name != st.session_state.pdf_name:
        st.session_state.messages = []
        st.session_state.pdf_name = uploaded_file.name

    chain, chunk_count = build_chain(
        uploaded_file.read(),
        uploaded_file.name
    )
    st.session_state.chain = chain
    st.success(f"✓ {uploaded_file.name} loaded — {chunk_count} chunks indexed")

# ─── Chat interface ───────────────────────────────────────────
if st.session_state.chain:
    st.divider()

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    if question := st.chat_input("Ask a question about your document..."):
        # show user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        # get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"query": question})
                answer = result["result"]
                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("👆 Upload a PDF to get started")