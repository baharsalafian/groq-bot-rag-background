import os
import streamlit as st
from groq import Groq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

# -----------------------
# Initialize session state
# -----------------------
if "docs" not in st.session_state:
    st.session_state.docs = []

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Groq API Setup (hardcoded key)
# -----------------------
groq_api_key = "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"  # << your API key here

if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing! Please provide a valid key.")
    st.stop()

client = Groq(api_key=groq_api_key)

# -----------------------
# Sidebar: Model Selection
# -----------------------
SUPPORTED_MODELS = {
    "Llama3-70b-8192": "llama3-70b-8192",
    "Mixtral-8x7b-32768": "mixtral-8x7b-32768",
    "Gemma-7b-It": "gemma-7b-it",
}
st.sidebar.title("Personalization")
model_friendly = st.sidebar.selectbox("Choose a model", list(SUPPORTED_MODELS.keys()))
model = SUPPORTED_MODELS[model_friendly]

# -----------------------
# Custom Title (one line, no wrap)
# -----------------------
st.markdown(
    """
    <h1 style='white-space: nowrap; font-size: 2.5em; font-weight: bold; text-align: center;'>
        💬 Ask Me Anything about Bahareh Salafian
    </h1>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Preload Resume Document
# -----------------------
if not st.session_state.docs:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(base_dir, "Bahareh Salafian Resume.pdf")

    if not os.path.exists(resume_path):
        st.error(f"❌ Resume file not found at {resume_path}. Please include it in the repo.")
        st.stop()

    if resume_path.endswith(".txt"):
        loader = TextLoader(resume_path, encoding="utf-8")
    elif resume_path.endswith(".pdf"):
        loader = PyPDFLoader(resume_path)
    elif resume_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(resume_path)
    else:
        st.error("Unsupported resume file format!")
        loader = None

    if loader:
        loaded_docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        for d in loaded_docs:
            st.session_state.docs.extend(text_splitter.split_text(d.page_content))

# -----------------------
# Render chat history
# -----------------------
for message in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])

# -----------------------
# User input (chat)
# -----------------------
if prompt := st.chat_input("Ask me anything about my background:"):
    scholar_link = "https://scholar.google.com/citations?user=qDsiKcIAAAAJ&hl=en"

    context_text = "\n".join(st.session_state.docs[:3])
    final_prompt = f"""Answer the question based on the following context:

Resume context:
{context_text}

Google Scholar: {scholar_link}

Question: {prompt}"""

    st.session_state.history.append({"query": prompt, "response": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model=model,
        )
        response = chat_completion.choices[0].message.content
    except Exception as e:
        response = f"⚠️ There was an error calling the model:\n\n{e}"

    st.session_state.history[-1]["response"] = response
    with st.chat_message("assistant"):
        st.markdown(response)
