import os
import json
import streamlit as st
from groq import Groq
from scholarly import scholarly  # For Google Scholar scraping

# Text splitter from main langchain
from langchain.text_splitter import CharacterTextSplitter

# Document loaders, embeddings, vectorstores from langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document  # For adding metadata


# -----------------------
# Initialize session state
# -----------------------
if "docs" not in st.session_state:
    st.session_state.docs = []

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Groq API Setup
# -----------------------
groq_api_key = st.secrets.get("GROQ_API_KEY", None)
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Please set it in Streamlit secrets.")
    st.stop()
client = Groq(api_key=groq_api_key)

# -----------------------
# Load Resume + LinkedIn PDF + Scholar Data (with tags)
# -----------------------
if not st.session_state.docs:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    # ✅ Load Resume
    resume_path = os.path.join(base_dir, "Bahareh Salafian Resume.pdf")
    if not os.path.exists(resume_path):
        st.error(f"❌ Resume file not found at {resume_path}")
        st.stop()

    resume_loader = PyPDFLoader(resume_path)
    resume_docs = resume_loader.load()
    for d in resume_docs:
        chunks = text_splitter.split_text(d.page_content)
        for c in chunks:
            st.session_state.docs.append(Document(page_content=c, metadata={"source": "resume"}))

    # ✅ Load LinkedIn PDF if available
    linkedin_path = os.path.join(base_dir, "linkedin_profile.pdf")
    if os.path.exists(linkedin_path):
        try:
            linkedin_loader = PyPDFLoader(linkedin_path)
            linkedin_docs = linkedin_loader.load()
            for d in linkedin_docs:
                chunks = text_splitter.split_text(d.page_content)
                for c in chunks:
                    st.session_state.docs.append(Document(page_content=c, metadata={"source": "linkedin"}))
            st.info("✅ LinkedIn profile PDF loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load LinkedIn PDF: {e}")
    else:
        st.info("ℹ️ LinkedIn profile PDF not found. Place 'linkedin_profile.pdf' in the same folder to include it.")

    # ✅ Add Google Scholar profile + fetch publications dynamically
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
# Create embeddings + FAISS
# -----------------------
if "vectorstore" not in st.session_state:
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.from_documents(st.session_state.docs, embedding_model)

# -----------------------
# Sidebar: Model selection
# -----------------------
st.sidebar.title("Personalization")
available_models = ["llama-3.3-70b-versatile"]
model = st.sidebar.selectbox("Choose a model", options=available_models)

# -----------------------
# Custom Title
# -----------------------
st.markdown(
    "<h1 style='text-align:center;'>💬 Ask Me Anything about Bahareh Salafian</h1>",
    unsafe_allow_html=True
)

# -----------------------
# Multi-turn chat & retrieval
# -----------------------
if prompt := st.chat_input("Ask me anything about my background:"):
    # Semantic retrieval
    docs = st.session_state.vectorstore.similarity_search(prompt, k=3)

    # Combine context with source tags
    context_text = "\n".join([f"[Source: {d.metadata['source']}] {d.page_content}" for d in docs])

    # Multi-turn context (last 3 turns)
    N = 3
    history_context = ""
    if st.session_state.history:
        last_turns = st.session_state.history[-N:]
        for turn in last_turns:
            history_context += f"User: {turn['query']}\nAssistant: {turn['response']}\n"

    # Final prompt for LLM
    final_prompt = f"""Answer the question based on the following context.
Always mention the source when relevant (Resume, LinkedIn, or Google Scholar).

Context:
{context_text}

Conversation history:
{history_context}

Question:
{prompt}"""

    # Call Groq LLM
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
        "feedback": None,
        "feedback_requested": False
    })

# -----------------------
# Render chat history with conditional feedback
# -----------------------
for i, message in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])

        if not message["feedback_requested"]:
            if st.button("✅ I'm done with my questions", key=f"done_{i}"):
                st.session_state.history[i]["feedback_requested"] = True

        if message["feedback_requested"]:
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
# Save feedback to file
# -----------------------
def save_feedback():
    with open("feedback.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

save_feedback()
