import os
import streamlit as st
import wikipedia
from pydantic import BaseModel
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.chains import RetrievalQA

# ---------- Setup ----------
API_KEY = "J080mnAHaVgnWNriS5kuwLEGhD6EpPHgRFovjLqP"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Unified AI Chatbot", layout="wide")
st.title("🤖 Unified AI Chatbot: Wikipedia | IPC | Skill India | Digital India")

# ---------- 1. Institution Info from Wikipedia ----------
st.header("🏫 Institution Info from Wikipedia")
institution = st.text_input("Enter Institution Name:")

if st.button("Fetch Institution Info"):
    if institution:
        try:
            page = wikipedia.page(institution)
            content = page.content.lower()
            summary = wikipedia.summary(institution, sentences=4)

            def extract_info(keyword):
                for line in content.split('\n'):
                    if keyword in line:
                        return line.strip()
                return "Not found"

            class InstitutionDetails(BaseModel):
                founder: str
                founded: str
                summary: str

            details = InstitutionDetails(
                founder=extract_info("founder"),
                founded=extract_info("founded"),
                summary=summary
            )

            st.subheader(f"📘 Information for '{institution}'")
            for field, value in details.model_dump().items():
                st.markdown(f"**{field.capitalize()}:** {value}")

        except wikipedia.exceptions.DisambiguationError as e:
            st.error("⚠️ Too many results. Try one of these:")
            for option in e.options[:5]:
                st.write("-", option)
        except wikipedia.exceptions.PageError:
            st.error("❌ Institution not found on Wikipedia.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter an institution name.")

# ---------- Helper: Format Answer as Bullet Points ----------
def format_answer(text):
    if isinstance(text, dict):
        text = text.get("result", "")
    if not text:
        return "No answer found."
    lines = text.split("\n")
    formatted = ""
    for line in lines:
        if line.strip().startswith("-"):
            formatted += f"- {line.strip()[1:].strip()}\n"
        elif line.strip():
            formatted += f"- {line.strip()}\n"
    return formatted

# ---------- 2. IPC Chatbot ----------
st.markdown("---")
st.header("📘 Indian Penal Code Chatbot")

ipc_question = st.text_input("Ask your IPC-related question:", key="ipc")

if ipc_question:
    try:
        docs = TextLoader("ipc.txt", encoding="utf-8").load()
        chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

        embedding = CohereEmbeddings(cohere_api_key=API_KEY, model="embed-english-v3.0")
        retriever = FAISS.from_documents(chunks, embedding).as_retriever()

        llm = ChatCohere(cohere_api_key=API_KEY, model="command-r-plus")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        response = qa.invoke(ipc_question)
        answer = format_answer(response)

        st.markdown(f"🔍 **You asked:** {ipc_question}")
        st.markdown("📘 **Answer:**")
        st.markdown(answer)
    except Exception as e:
        st.error(f"❌ Error in IPC chatbot: {e}")
else:
    st.info("Ask a question related to the Indian Penal Code.")

# ---------- 3. Digital India Chatbot ----------
st.markdown("---")
st.header("🌐 Digital India Document Chatbot")

digital_question = st.text_input("Ask something about Digital India:", key="digitalindia")

if digital_question:
    try:
        if not os.path.exists("digital_india.txt"):
            st.error("⚠️ digital_india.txt not found. Please place it in the app folder.")
        else:
            docs = TextLoader("digital_india.txt", encoding="utf-8").load()
            chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

            embedding = CohereEmbeddings(cohere_api_key=API_KEY, model="embed-english-v3.0")
            retriever = FAISS.from_documents(chunks, embedding).as_retriever()

            llm = ChatCohere(cohere_api_key=API_KEY, model="command-r-plus")
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            response = qa.invoke(digital_question)
            answer = format_answer(response)

            st.markdown(f"🔍 **You asked:** {digital_question}")
            st.markdown("📘 **Answer:**")
            st.markdown(answer)
    except Exception as e:
        st.error(f"❌ Error in Digital India chatbot: {e}")
else:
    st.info("Ask anything about the Digital India programme based on the provided document.")
