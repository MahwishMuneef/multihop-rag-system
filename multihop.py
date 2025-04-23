import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma as ChromaClient
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions

# Suppress Streamlit missing ScriptRunContext warnings
logging.getLogger('streamlit.runtime.scriptrunner_utils').setLevel(logging.ERROR)

# Load API keys
def load_api_key():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_azure_safety_client():
    endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
    key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
    return ContentSafetyClient(endpoint, AzureKeyCredential(key))

load_api_key()
safety_client = load_azure_safety_client()

# Content Safety Check
def is_content_safe(text: str, client: ContentSafetyClient) -> bool:
    try:
        request = AnalyzeTextOptions(text=text)
        response = client.analyze_text(request)
        for category in response.categories_analysis:
            print(f"[Category: {category.category}] Severity: {category.severity}")
            if category.severity > 1:
                return False
        return True
    except Exception as e:
        print("Content Safety check failed:", e)
        return True


pdf_paths = [
    "/Users/mahwishmuneef/Desktop/ Gen AI/data/CMAM Guidelines.pdf",
    "/Users/mahwishmuneef/Desktop/ Gen AI/data/EWEC_globalstrategyreport_200915_FINAL_WEB.pdf",
    "/Users/mahwishmuneef/Desktop/ Gen AI/data/Maternal mortality.pdf",
    "/Users/mahwishmuneef/Desktop/ Gen AI/data/Maternal Neonatal Child Health : Midwifery – Midwifery Association of Pakistan.pdf",
    "/Users/mahwishmuneef/Desktop/ Gen AI/data/Pakistan Maternal Nutrition Strategy 2022-27.pdf"
]


embedding = OpenAIEmbeddings()

if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
    st.sidebar.info("Initializing Chroma DB from local PDFs...")
    os.makedirs("./chroma_db", exist_ok=True)
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(all_docs)
    ChromaClient.from_documents(docs, embedding, persist_directory="./chroma_db")

db = ChromaClient(persist_directory="./chroma_db", embedding_function=embedding)
retriever = db.as_retriever()
llm = OpenAI(temperature=0)

def select_llm_by_context_length(text: str):
    num_tokens = len(text) // 4
    if num_tokens < 3000:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    elif num_tokens < 7500:
        return ChatOpenAI(model="gpt-4", temperature=0)
    else:
        st.warning("Context too long! Truncating to fit GPT-4...")
        return ChatOpenAI(model="gpt-4", temperature=0)

def decompose_question(question: str) -> list:
    prompt = f"""
You are a helpful assistant. Decompose into **precise sub-questions** aimed at retrieving **specific facts or guidance** that would help answer the original query.

Original Question:
"{question}"

Sub-questions:
1.
2.
3.
"""
    result = llm.invoke(prompt)
    return [line.strip()[3:] for line in result.split("\n") if line.strip().startswith(("1.", "2.", "3."))]

def get_combined_context(sub_questions, retriever_func):
    docs = []
    for sub_q in sub_questions:
        candidates = (
            retriever_func.invoke(sub_q)
            if hasattr(retriever_func, 'invoke')
            else retriever_func.get_relevant_documents(sub_q)
        )
        docs.extend(candidates[:3])
    return "\n\n".join([d.page_content for d in docs])

def answer_question(original_query, retriever_func):
    subs = decompose_question(original_query)
    context = get_combined_context(subs, retriever_func)
    llm_fin = select_llm_by_context_length(context)
    prompt = f"""
You are a maternal health expert.

Answer the question below using only the provided context. Be concise and informative — keep your answer under 10 lines.

Question:
"{original_query}"

Context:
{context}
"""
    return llm_fin.invoke(prompt)

def answer_multi_hop_query(q, retriever_func=retriever):
    return answer_question(q, retriever_func)

def run_streamlit_app():
    st.title("Maternal Health Multi-Hop QA")
    st.sidebar.header("Upload Document for QA")
    file = st.sidebar.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"] )

    if file:
        os.makedirs("./uploaded_docs", exist_ok=True)
        path = os.path.join("./uploaded_docs", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        if file.type == "application/pdf":
            loader = PyPDFLoader(path)
            docs = loader.load_and_split()
        else:
            text = open(path, encoding="utf-8").read()
            parts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_text(text)
            docs = [Document(page_content=p) for p in parts]
        temp_db = ChromaClient.from_documents(
            documents=docs,
            embedding=embedding
        )
        retriever_func = temp_db.as_retriever()
        st.sidebar.success("Indexed uploaded document.")
    else:
        retriever_func = retriever

    q = st.text_input("Enter your question:")

    if q:
        if not is_content_safe(q, safety_client):
            st.subheader("⚠️ Unsafe Question Detected")
            st.error("Your question may contain harmful or inappropriate content. Please rephrase it.")
        else:
            with st.spinner("Generating answer..."):
                out = answer_multi_hop_query(q, retriever_func)
                answer_text = out.content if hasattr(out, "content") else str(out)

            if not is_content_safe(answer_text, safety_client):
                st.subheader("⚠️ Unsafe Content Detected")
                st.error("The generated content may be harmful or inappropriate.")
            else:
                st.subheader("Answer")
                st.write(answer_text)


if __name__ == "__main__":
    run_streamlit_app()