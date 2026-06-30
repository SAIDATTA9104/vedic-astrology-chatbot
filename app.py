import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables (GROQ_API_KEY should be present in .env)
load_dotenv()

st.set_page_config(page_title="Astrological Assistant", page_icon="✨")

# Minimalist title
st.title("✨ Astrological Assistant")

# The specific system prompt provided in instructions
SYSTEM_PROMPT = """You are an expert, analytical Vedic Astrologer (Jyotishi) acting as a data-driven interpreter.
Your objective is to answer the user's question by meticulously analyzing the provided Parashara's Light (PL9) astrological report data.

### 1. STRICT GROUNDING RULES
- You must rely EXCLUSIVELY on the data provided in the `<context>` block. 
- Do NOT hallucinate planetary placements, Dasha periods, or Yogas. If the context does not explicitly contain the data required to answer the question, state: "The provided chart segment does not contain the required data for this specific query."
- Treat the PL9 numerical data (Shadbala, Ashtakavarga scores, degrees, exact dates) as immutable facts.

### 2. ANALYTICAL FRAMEWORK
When synthesizing an answer, use the following logical progression to ensure astrological accuracy:
1. Base Placement: Identify the relevant planet(s) related to the query, its house, and its sign.
2. Strengths & Dignity: Evaluate its condition using any provided Shadbala scores, dignity (exalted/debilitated), or Varga chart placements (like Navamsha/D9).
3. Influences: Factor in aspects (Drishti) and conjunctions from natural benefics (Jupiter, Venus) or malefics (Saturn, Mars, Rahu, Ketu).
4. Timing: Cross-reference the analysis with the currently active Vimshottari Dasha (Mahadasha, Antardasha) timelines provided in the text.

### 3. TONE AND OUTPUT FORMAT
- Tone: Be objective, precise, and practical. Avoid fatalistic, doom-mongering, or overly mystical language.
- Phrasing: Frame predictions as "strong indications," "karmic tendencies," or "periods of specific focus." 
- Structure: Break down your analysis logically. Use bullet points for readability. Always bold specific dates, planetary names, and Dasha periods.
- Do not prescribe generalized remedies or gemstones unless they are explicitly recommended within the context document.

<context>
{context}
</context>"""

@st.cache_resource(show_spinner=False)
def load_and_initialize_rag():
    """
    Ingest local PDF and TXT files from the astrology_reports directory,
    embed them, and return a conversational RAG chain.
    """
    reports_dir = "./astrology_reports/"
    
    # Ensure directory exists
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir, exist_ok=True)
        return None

    # We use a subtle loading spinner during startup ingestion
    with st.spinner("Initializing local knowledge base..."):
        # Load PDFs using PyPDFLoader via DirectoryLoader
        pdf_loader = DirectoryLoader(
            reports_dir, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader,
            use_multithreading=True
        )
        docs = pdf_loader.load()
        
        # Load text files
        txt_loader = DirectoryLoader(reports_dir, glob="**/*.txt")
        docs.extend(txt_loader.load())

        if not docs:
            # If the directory is completely empty, we return None to inform the user
            return None

        # Chunk the documents to fit into the LLM context window
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # NOTE: Groq currently does not provide a native embedding API.
        # We are using a robust, lightweight local embedding model instead 
        # (sentence-transformers/all-MiniLM-L6-v2) which is standard for local RAG.
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}, # Change to 'cuda' or 'mps' if you have a GPU
    encode_kwargs={'normalize_embeddings': True}
            )

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Initialize the core LLM generation model via Hugging Face API
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            temperature=0.6,
            max_new_tokens=2048
        )
        llm = ChatHuggingFace(llm=llm_endpoint)

        # Set up retrieval chain with history awareness to handle follow-up questions properly
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain

# Initialize the pipeline and vector database on startup
rag_chain = load_and_initialize_rag()

if rag_chain is None:
    st.info("No reports found. Please drop your PDF or TXT files into the `./astrology_reports/` directory and restart the application.")
else:
    # Initialize session state for UI chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize session state for Langchain conversation memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box at the bottom of the screen
    if prompt := st.chat_input("Ask a question about your astrological report..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing reports..."):
                try:
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Store messages in session memory
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    # Update LangChain memory format
                    st.session_state.chat_history.extend([
                        HumanMessage(content=prompt),
                        AIMessage(content=answer)
                    ])
                except Exception as e:
                    st.error(f"Error querying the model: {str(e)}")