# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming

# Import necessary libraries for the application
import os  # Provides utilities to interact with the operating system (e.g., file paths).
import base64  # Used to encode binary files like PDFs for embedding in HTML.
import gc  # Python's garbage collector to free up memory.
import random  # Generates random values (currently unused).
import tempfile  # Creates temporary files and directories.
import time  # Provides time-related utilities (e.g., delays).
import uuid  # Generates unique identifiers (used to maintain session state).

# For rendering Markdown content (replaceable with Streamlit's native markdown support if desired)
from IPython.display import Markdown, display

# Libraries for LLM-powered document processing and querying
from llama_index.core import Settings  # Manages settings for the Llama Index library.
from llama_index.llms.ollama import Ollama  # Interface to load and use the Ollama LLM.
from llama_index.core import PromptTemplate  # To customize query prompts for the LLM.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace embedding model.
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader  # Core utilities for indexing and document handling.

# Streamlit library for building the web app
import streamlit as st

# Initialize session state variables if not already present
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()  # Generate a unique identifier for the session.
    st.session_state.file_cache = {}  # Cache to store processed files and avoid reprocessing.

# Save session ID for easier reference throughout the app
session_id = st.session_state.id
client = None  # Placeholder for future client integrations.

# Load the LLM with caching to avoid redundant resource consumption
@st.cache_resource
def load_llm():
    # Loads the specified model with a timeout of 120 seconds
    llm = Ollama(model="llama3.2:1b", request_timeout=120.0)
    return llm

# Function to reset the chat context and messages
def reset_chat():
    st.session_state.messages = []  # Clear the chat history.
    st.session_state.context = None  # Reset any context stored for the LLM.
    gc.collect()  # Trigger garbage collection to free up memory.

# Function to display an uploaded PDF file in the app
def display_pdf(file):
    st.markdown("### PDF Preview")  # Display a heading for the PDF preview section.
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")  # Encode the PDF in Base64 format.
    
    # Embed the Base64-encoded PDF in an HTML <iframe>
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" 
            type="application/pdf" style="height:100vh; width:100%">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)  # Render the embedded PDF.

# Sidebar setup for document uploading
with st.sidebar:
    st.header(f"Add your documents!")  # Sidebar header.
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")  # File uploader for PDFs.

    # Process the uploaded file if one is provided
    if uploaded_file:
        try:
            # Create a temporary directory to store the uploaded file
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)  # Generate file path.
                
                # Save the uploaded file locally within the temporary directory
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"  # Generate a unique key for the file.

                # Check if the file is already cached
                if file_key not in st.session_state.get('file_cache', {}):
                    # Load documents from the temporary directory
                    if os.path.exists(temp_dir):
                        loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf"], recursive=True)
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    docs = loader.load_data()  # Load data from the documents.

                    # Load the LLM and embedding model
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    Settings.embed_model = embed_model  # Set the embedding model in the settings.

                    # Create a vector index for the loaded documents
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)
                    Settings.llm = llm  # Set the LLM in the settings.

                    # Create a query engine for interacting with the indexed documents
                    query_engine = index.as_query_engine(streaming=True)

                    # Customize the query prompt
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query "
                        "in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

                    # Cache the query engine for the file
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]  # Retrieve from cache if already processed.

                # Inform the user and display the uploaded PDF
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Header and reset button layout
col1, col2 = st.columns([6, 1])
with col1:
    st.header(f"Chat with Docs using Llama-3.2")  # Main header for the app.
with col2:
    st.button("Clear ↺", on_click=reset_chat)  # Button to clear chat history.

# Initialize chat history if not already present
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # Render past chat messages.

# Handle new user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})  # Save user message.
    with st.chat_message("user"):
        st.markdown(prompt)  # Display the user's input.

    # Process the query and display assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder for streaming response.
        full_response = ""
        
        # Get response in chunks and display it incrementally
        streaming_response = query_engine.query(prompt)
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")  # Stream response with cursor effect.

        message_placeholder.markdown(full_response)  # Display the full response.
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})  # Save assistant's response.
