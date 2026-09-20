import streamlit as st

from rag.config import (
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    TOP_K
)

from rag.retriever import GovernmentRetriever
from rag.llm import GovernmentLLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Government AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🏛️ Government AI Knowledge Assistant
    </div>

    <div class="subtitle">
        Search and ask questions across
        multiple government departments.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for position, item in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
==================================================
CONTEXT {position}
==================================================

Department:
{item["department"]}

Source:
{item["source"]}

Page:
{item["page"]}

Chunk:
{item.get("chunk_on_page", "N/A")}

Global Chunk Order:
{item.get("global_order", "N/A")}

Content:
{item["text"]}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# LOAD RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    return GovernmentRetriever(
        VECTORSTORE_PATH,
        EMBEDDING_MODEL
    )


retriever = load_retriever()


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    return GovernmentLLM(
        st.secrets["GROQ_API_KEY"]
    )


llm = load_llm()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    department = st.selectbox(
        "Department",
        [
            "All Departments",
            "Finance",
            "Education",
            "Health",
            "Interior",
            "Agriculture"
        ]
    )

    top_k = st.slider(
        "Retrieved Documents",
        min_value=3,
        max_value=10,
        value=TOP_K
    )

    st.divider()

    st.caption(
        "Powered by FAISS + "
        "Sentence Transformers + "
        "Groq GPT-OSS 120B"
    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about government departments..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # -------------------------------
    # USER MESSAGE
    # -------------------------------

    with st.chat_message("user"):

        st.write(question)


    # -------------------------------
    # RETRIEVAL
    # -------------------------------

    with st.spinner(
        "Searching government documents..."
    ):

        selected_department = None

        if department != "All Departments":

            selected_department = department

        results = retriever.search(
            question,
            top_k=top_k,
            department=selected_department
        )


    # -------------------------------
    # NO RESULTS
    # -------------------------------

    if not results:

        with st.chat_message("assistant"):

            st.warning(
                "No relevant government documents "
                "were found for this question."
            )

    else:

        # -------------------------------
        # BUILD CONTEXT
        # -------------------------------

        context = build_context(
            results
        )


        # -------------------------------
        # GENERATE ANSWER
        # -------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = llm.generate(
                question,
                context
            )


        # -------------------------------
        # ASSISTANT ANSWER
        # -------------------------------

        with st.chat_message("assistant"):

            st.markdown(answer)


        # -------------------------------
        # SOURCES
        # -------------------------------

        st.subheader("📚 Sources")

        for item in results:

            with st.expander(
                f"{item['department']} — "
                f"{item['source']} "
                f"(Page {item['page']})"
            ):

                st.write(
                    item["text"]
                )

                st.caption(
                    f"Similarity: "
                    f"{item['score']:.4f}"
                )