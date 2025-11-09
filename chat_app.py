import os
import traceback
import streamlit as st

# --------------------------
# Configuration - NO DOTENV NEEDED
# --------------------------
try:
    # Try to get from Streamlit secrets (for deployment)
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    COHERE_MODEL = st.secrets.get("COHERE_MODEL", "command-a-03-2025")
except:
    # Fallback to environment variables or hardcoded (for local development)
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "Hx1jh8VxQfz0fj2IDCdbhG8RH913X8ifDoCnAVjz")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
    COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-03-2025")

# --------------------------
# Initialize API Clients
# --------------------------
@st.cache_resource
def init_clients():
    try:
        import cohere
        from tavily import TavilyClient
    except ImportError as e:
        st.error("Missing packages: install with `pip install cohere tavily-python`")
        raise

    if not COHERE_API_KEY or not TAVILY_API_KEY:
        st.error("Missing API keys. Please check your configuration.")
        st.info("For local development: Set COHERE_API_KEY and TAVILY_API_KEY in your environment")
        st.info("For deployment: Add them to Streamlit Cloud secrets")
        st.stop()

    try:
        co = cohere.Client(COHERE_API_KEY)
        tv = TavilyClient(api_key=TAVILY_API_KEY)
        return co, tv
    except Exception as e:
        st.error(f"Failed to initialize API clients: {str(e)}")
        return None, None

# --------------------------
# Main Research Logic
# --------------------------
def search_and_answer(question, co_client, tv_client):
    if not co_client or not tv_client:
        return "Missing API clients. Check API keys.", []

    try:
        # Step 1: Search the web
        search_result = tv_client.search(query=question, search_depth="basic", max_results=5)
        sources = search_result.get("results", []) if isinstance(search_result, dict) else []

        if not sources:
            return "No information found on this topic.", []

        # Step 2: Prepare search context
        context_parts = []
        for i, s in enumerate(sources, start=1):
            title = s.get("title", "No title").strip()
            url = s.get("url", "").strip()
            content = s.get("content", "").replace("\n", " ").strip()[:800]
            context_parts.append(f"[{i}] {title}\nURL: {url}\n{content}\n")

        context = "\n\n".join(context_parts)

        # Step 3: Prompt for Cohere Chat API
        prompt = (
            f"Answer the question: {question}\n\n"
            "Use only the following sources and cite them inline as [1], [2], etc.\n\n"
            f"{context}\n\n"
            "Requirements:\n"
            "- Be factual\n"
            "- Cite sources using [n]\n"
            "- Include a 'Sources' section with title and URL\n"
        )

        # Step 4: Call Cohere Chat API
        resp = co_client.chat(model=COHERE_MODEL, message=prompt, temperature=0.3)

        text = getattr(resp, "text", str(resp))
        return text, sources

    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}", []

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.set_page_config(page_title="AI Research Assistant", layout="wide")
    st.title("AI Research Assistant")
    st.write("Ask a question — I'll search the web and generate an answer with citations.")

    if "history" not in st.session_state:
        st.session_state.history = []

    co_client, tv_client = init_clients()

    # Use chat input for better UX
    question = st.chat_input("Your research question")

    if question and question.strip():
        with st.spinner("Searching and generating answer..."):
            answer, sources = search_and_answer(question, co_client, tv_client)
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

    # Display chat history
    for i, item in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(item["question"])
        
        with st.chat_message("assistant"):
            st.write(item['answer'])
            if item["sources"]:
                with st.expander("Sources"):
                    for j, s in enumerate(item["sources"], start=1):
                        st.markdown(f"**[{j}] {s.get('title', 'No title')}**")
                        if s.get("url"):
                            st.markdown(f"Link: {s['url']}")
                        if s.get("content"):
                            st.markdown(f"{s.get('content', '')[:200]}...")
                        st.markdown("---")

if __name__ == "__main__":
    main()
