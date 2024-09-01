import streamlit as st
import os
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set environment variable for macOS to avoid potential issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set page configuration
st.set_page_config(page_title="Document Retrieval Chatbot", page_icon=":robot_face:", layout="wide")

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  
GROQ_API_KEY = "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"  # Replace with your Groq API key

# Function to read documents from a directory
def read_documents(directory):
    file_loader = PyPDFDirectoryLoader(directory, extract_images=False)
    documents = file_loader.load()
    return documents

# Function to chunk documents
def chunk_documents(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

# Load a pre-trained model from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L12-v2"  # Can even use a larger model for better embeddings but speed will be compromised.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate document embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Initialize FAISS index
docs = read_documents('documents/')
documents = chunk_documents(docs=docs)

embedding_dim = generate_embedding("test").shape[0]
index = faiss.IndexFlatL2(embedding_dim)

# Function to add documents to the FAISS index
def add_documents_to_faiss(documents):
    for doc in documents:
        try:
            doc_vector = generate_embedding(doc.page_content)
            index.add(doc_vector.reshape(1, -1))
        except Exception as e:
            st.error(f"Error while adding document: {e}")
    st.success("Documents added to FAISS index.")

add_documents_to_faiss(documents)

# Function to retrieve documents based on a query from FAISS
def retrieve_documents(query, k=2):
    query_embedding = generate_embedding(query).reshape(1, -1)
    D, I = index.search(query_embedding, k=k)
    matching_docs = [documents[i] for i in I[0]]
    return matching_docs

# Function to generate a conversational response using Groq API
def generate_conversational_response(query, documents):
    context = " ".join([doc.page_content for doc in documents])
    summarized_context = summarize_text(context)
    
    input_text = f"User query: {query}\nRelevant information: {summarized_context}\nProvide a detailed summary or a conversational response."
    response = call_groq_api(input_text)
    
    return response

# Function to summarize text using Groq API
def summarize_text(text):
    payload = {
        "model": "llama3-8b-8192",  # Use the appropriate model ID as required
        "messages": [{"role": "system", "content": f"Summarize this: {text}"}],
        "max_tokens": 150  # Adjust max tokens for approx 100-word summary
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Check for HTTP errors
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Groq API: {e}")
        return ""  # Return empty string on failure

# Function to call Groq API
def call_groq_api(text):
    payload = {
        "model": "llama3-8b-8192",  # Use the appropriate model ID as per your requirement.
        "messages": [{"role": "system", "content": text}],
        "max_tokens": 300  # Increase max tokens for more detailed response
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Check  status for HTTP errors
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Groq API: {e}")
        return ""  # Return empty string on failure

# Streamlit app with chatbot interface
def main():
    st.title("📄 Document Retrieval Chatbot")
    st.sidebar.header("Chatbot Configuration")

    # Initialize session state for conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # User input for query
    query = st.sidebar.text_input("Ask a question:")

    if st.sidebar.button("Send"):
        if query:
            st.session_state.history.append((query, ""))  # Add user query to history

            # Retrieve matching documents
            matching_docs = retrieve_documents(query)
            
            # Generate a conversational response
            bot_response = generate_conversational_response(query, matching_docs)

            st.session_state.history[-1] = (query, bot_response)  # Update history with bot response

            # Display bot response
            st.sidebar.markdown("**User:** " + query)
            st.sidebar.markdown("**Bot:** " + bot_response)
        else:
            st.sidebar.warning("Please enter a question.")
    
    # Display conversation history
    st.subheader("Conversation History")
    with st.expander("Show/Hide History"):
        for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
            st.markdown(f"**User:** {user_msg}")
            st.markdown(f"**Bot:** {bot_msg}")

if __name__ == "__main__":
    main()
