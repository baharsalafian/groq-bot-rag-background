import os
import streamlit as st
from groq import Groq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import json

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
# Load Resume & Split
# -----------------------
if not st.session_state.docs:
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
            st.session_state.docs.extend(text_splitter.split_text(d.page_content))

# -----------------------
# Create embeddings + FAISS
# -----------------------
if "vectorstore" not in st.session_state:
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.from_texts(st.session_state.docs, embedding_model)

# -----------------------
# Sidebar: model selection
# -----------------------
st.sidebar.title("Personalization")
available_models = ["llama-3.3-70b-versatile"]
model = st.sidebar.selectbox("Choose a model", options=available_models)

# -----------------------
# Custom title
# -----------------------
st.markdown("<h1 style='text-align:center;'>💬 Ask Me Anything about Bahareh Salafian</h1>", unsafe_allow_html=True)

# -----------------------
# Multi-turn chat & retrieval
# -----------------------
if prompt := st.chat_input("Ask me anything about my background:"):
    # Semantic retrieval
    docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
    context_text = "\n".join([d.page_content for d in docs])

    # Multi-turn context
    N = 3
    history_context = ""
    if st.session_state.history:
        last_turns = st.session_state.history[-N:]
        for turn in last_turns:
            history_context += f"User: {turn['query']}\nAssistant: {turn['response']}\n"

    # Final prompt
    final_prompt = f"""Answer the question based on the following context:

Resume context:
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

    # Save response with feedback placeholder & "feedback_requested" flag
    st.session_state.history.append({
        "query": prompt,
        "response": response,
        "feedback": None,
        "feedback_requested": False  # feedback will only be shown after user confirms
    })

# -----------------------
# Render chat history with conditional feedback
# -----------------------
for i, message in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])

        # Ask user if they are done with the answer
        if not message["feedback_requested"]:
            if st.button("✅ I'm done with this answer", key=f"done_{i}"):
                st.session_state.history[i]["feedback_requested"] = True

        # Show feedback buttons only after confirmation
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
# Optional: Save feedback to file
# -----------------------
def save_feedback():
    with open("feedback.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

save_feedback()
