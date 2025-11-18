# AmbedkarGPT
import os
import sys
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# --- Configuration Constants ---
TEXT_FILE = "speech.txt"
VECTOR_DB_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
CHROMA_COLLECTION_NAME = "ambedkar_speech_qa"

def setup_rag_pipeline():
    """
    Initializes and sets up the complete RAG pipeline:
    1. Loads the document.
    2. Splits the document into chunks.
    3. Creates embeddings and stores them in ChromaDB.
    4. Initializes the Ollama LLM and the RAG chain.
    """
    print("--- RAG Pipeline Setup Initiated ---")

    # 1. Load the provided text file (speech.txt)
    print(f"1. Loading document from: {TEXT_FILE}")
    if not os.path.exists(TEXT_FILE):
        print(f"Error: Required file '{TEXT_FILE}' not found.")
        print("Please create this file and paste the provided text into it.")
        sys.exit(1)

    try:
        loader = TextLoader(TEXT_FILE)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading document: {e}")
        sys.exit(1)

    # 2. Split the text into manageable chunks.
    # The text is very short, but we follow the pipeline requirement for robustness.
    print("2. Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,  # Small size appropriate for the short text
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    print(f"   -> Created {len(texts)} chunks.")

    # 3. Create Embeddings and store them in a local vector store (ChromaDB).
    print(f"3. Initializing HuggingFace Embeddings: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Use CPU to ensure 100% local operation
        )
    except Exception as e:
        print(f"Error initializing embeddings. Check internet connection for initial download: {e}")
        sys.exit(1)

    # Check if the vector store already exists to avoid re-embedding on every run
    if os.path.exists(VECTOR_DB_DIR):
        print(f"   -> Loading existing Chroma DB from {VECTOR_DB_DIR}")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
    else:
        print(f"   -> Creating and persisting new Chroma DB to {VECTOR_DB_DIR}")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )
        vectorstore.persist()
        print("   -> Chroma DB created and persisted.")

    # 5. Initialize the Ollama LLM
    # Assumes Ollama is running locally and 'mistral' model is pulled.
    print(f"4. Initializing Ollama LLM with model: {OLLAMA_MODEL}")
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        # Simple test to verify connection
        # llm.invoke("Hi") 
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print("Ensure Ollama is installed, running, and the 'mistral' model is pulled.")
        sys.exit(1)

    # 5. Initialize the RetrievalQA chain
    # This chain automatically handles steps 4 (Retrieval) and 5 (Generation).
    print("5. Initializing RetrievalQA Chain.")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False # Set to True for debugging context retrieval
    )

    print("--- RAG Pipeline Setup Complete. Ready to Q&A. ---")
    return qa_chain

def main():
    """
    Main function to run the command-line Q&A loop.
    """
    qa_chain = setup_rag_pipeline()

    print("\n--- Start Q&A Session ---")
    print(f"Ask a question based on the content in '{TEXT_FILE}'.")
    print("Type 'exit' or 'quit' to end the session.")
    print("-------------------------\n")

    while True:
        try:
            # Get a user's question
            question = input("Your Question: ").strip()

            if question.lower() in ['exit', 'quit']:
                print("\nSession ended. Goodbye!")
                break
            
            if not question:
                continue

            # Generate an answer by feeding the retrieved context and the question to an LLM.
            print("Thinking...")
            # The QA chain handles steps 4 (retrieval) and 5 (generation)
            response = qa_chain.invoke({"query": question})
            M
            # Print the final answer
            answer = response.get('result', 'Could not generate an answer.')
            print("\n🤖 Answer:")
            print(f"{answer}\n")

        except KeyboardInterrupt:
            print("\nSession ended by user (Ctrl+C). Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()


    langchain==0.1.13
langchain-core==0.1.35
langchain-community==0.0.29
pydantic==2.7.1
chromadb==0.4.24
sentence-transformers==2.7.0
ollama==0.1.84 # Ensure the correct Ollama SDK version

Install Ollama: Follow the instructions on the Ollama website for your operating system.

Pull the Mistral Model: Open your terminal and run the following command to download the required LLM model:
ollama pull mistral

It is highly recommended to use a virtual environment.
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
Install python dependencies
pip install -r requirements.txt
