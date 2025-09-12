import os
import json
import streamlit as st
from groq import Groq
from scholarly import scholarly  # For Google Scholar scraping
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

# -----------------------
# Session state
# -----------------------
if "docs" not in st.session_state:
    st.session_state.docs = []
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Groq API
# -----------------------
groq_api_key = st.secrets.get("GROQ_API_KEY", None)
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Set it in Streamlit secrets.")
    st.stop()
client = Groq(api_key=groq_api_key)

# -----------------------
# Helper: Fetch Groq models safely
# -----------------------
@st.cache_data(ttl=3600)
def get_groq_models():
    try:
        groq_models = client.models.list()
        model_names = []
        for m in groq_models:
            if isinstance(m, dict) and "name" in m:
                model_names.append(m["name"])
            elif isinstance(m, (list, tuple)) and len(m) > 0:
                model_names.append(m[0])
        return model_names or ["llama-3.3-70b-versatile"]
    except Exception as e:
        st.warning(f"⚠️ Could not fetch models from Groq: {e}")
        return ["llama-3.3-70b-versatile"]

# -----------------------
# Load Resume + LinkedIn + Scholar
# -----------------------
if not st.session_state.docs:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Resume
    resume_path = os.path.join(base_dir, "Bahareh Salafian Resume.pdf")
    if not os.path.exists(resume_path):
        st.error(f"❌ Resume file not found at {resume_path}")
        st.stop()
    if resume_path.endswith(".txt"):
        loader = TextLoader(resume_path, encoding="utf-8")
    elif resume_path.endswith(".pdf"):
        loader = PyPDFLoader(resume_path)
    elif resume_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(resume_path)
    else:
        st.error("Unsupported resume format")
        loader = None
    if loader:
        loaded_docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        for d in loaded_docs:
            chunks = text_splitter.split_text(d.page_content)
            for c in chunks:
                st.session_state.docs.append(Document(page_content=c, metadata={"source": "resume"}))

    # LinkedIn PDF
    linkedin_path = os.path.join(base_dir, "linkedin_profile.pdf")
    if os.path.exists(linkedin_path):
        try:
            loader = PyPDFLoader(linkedin_path)
            docs = loader.load()
            for d in docs:
                chunks = text_splitter.split_text(d.page_content)
                for c in chunks:
                    st.session_state.docs.append(Document(page_content=c, metadata={"source": "linkedin"}))
            st.info("✅ LinkedIn profile PDF loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load LinkedIn PDF: {e}")

    # Google Scholar
    try:
        author = scholarly.search_author_id("qDsiKcIAAAAJ")
        author_filled = scholarly.fill(author, sections=["publications"])
        for pub in author_filled["publications"]:
            title = pub["bib"]["title"]
            year = pub["bib"].get("pub_year", "N/A")
            venue = pub["bib"].get("venue", "N/A")
            text = f"Publication: {title}, Year: {year}, Venue: {venue}"
            st.session_state.docs.append(Document(page_content=text, metadata={"source": "scholar"}))
        st.info("✅ Google Scholar publications loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch Google Scholar publications automatically: {e}")
        st.session_state.docs.append(
            Document(page_content="Google Scholar Profile: https://scholar.google.com/citations?user=qDsiKcIAAAAJ&hl=en",
                     metadata={"source": "scholar"})
        )

# -----------------------
# Embeddings + FAISS
# -----------------------
if "vectorstore" not in st.session_state:
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.from_documents(st.session_state.docs, embedding_model)

# -----------------------
# Sidebar: Model selection
# -----------------------
st.sidebar.title("Personalization")
available_models = get_groq_models()
model = st.sidebar.selectbox("Choose a model", options=available_models)

# -----------------------
# Custom title
# -----------------------
st.markdown("<h1 style='text-align:center;'>💬 Ask Me Anything about Bahareh Salafian</h1>", unsafe_allow_html=True)

# -----------------------
# Multi-turn chat & retrieval
# -----------------------
if prompt := st.chat_input("Ask me anything about my background:"):
    docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
    context_text = "\n".join([f"[Source: {d.metadata['source']}] {d.page_content}" for d in docs])

    N = 3  # last N turns
    history_context = ""
    if st.session_state.history:
        last_turns = st.session_state.history[-N:]
        for turn in last_turns:
            history_context += f"User: {turn['query']}\nAssistant: {turn['response']}\n"

    final_prompt = f"""Answer the question based on the following context:

Context:
{context_text}

Conversation history:
{history_context}

Question:
{prompt}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model=model
        )
        response = chat_completion.choices[0].message.content
    except Exception as e:
        response = f"⚠️ Error calling model: {e}"

    st.session_state.history.append({
        "query": prompt,
        "response": response,
        "feedback": None
    })

# -----------------------
# Render chat history + feedback buttons
# -----------------------
for i, message in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", key=f"up_{i}"):
                st.session_state.history[i]["feedback"] = "helpful"
                st.success("Feedback recorded!")
        with col2:
            if st.button("👎 Not Helpful", key=f"down_{i}"):
                st.session_state.history[i]["feedback"] = "not_helpful"
                st.error("Feedback recorded!")

# -----------------------
# Save feedback
# -----------------------
def save_feedback():
    with open("feedback.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

save_feedback()
