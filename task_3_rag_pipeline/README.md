# 🩺 Medical RAG - Extractive QA System

This project is a simple yet powerful 'Extraction' Question-Answering system using the Retrieval-Augmented Generation (RAG) architecture. It is built with Streamlit, SentenceTransformers, FAISS, and Hugging Face Transformers.

The application answers medical questions by first retrieving relevant text chunks from a local document corpus (your `.txt` files) and then using an extractive QA model to find the precise answer within that context.

The DistilBERT model said in the project instructions is meant for only Extraction tasks and not for building a Proper assistant, but still I've implemented a simple QA system here using it. If the choice was left to me, I'd rather use an API Service like Groq, which are way faster and fit the purpose really well.

---

## ✨ Features

-   **Extractive Question-Answering**: Extracts direct answers to your questions from the provided documents.
-   **Document-Based Context**: Uses `.txt` files in a `docs/` directory as its knowledge base.
-   **Transparent Context**: Shows the exact text chunks that were used to generate the answer.
-   **Interactive UI**: A simple and clean web interface powered by Streamlit.
-   **Efficient Search**: Utilizes FAISS for fast and efficient similarity search of document chunks.

---

## ⚙️ How It Works

The application follows a classic RAG pipeline:

1.  **Load Corpus**: All `.txt` files inside the `docs/` folder are loaded into a single corpus.For the push to GitHub exceeding the allowed repo size, I had to remove some text docs from it. NB: If you want to add any further documents to the list, do create more .txt files in the docs folder, you can copy it from any website such as Wikipedia or download .txt files from the web and place it there in the docs folder.
2.  **Chunking**: The corpus is split into smaller, overlapping text chunks using `RecursiveCharacterTextSplitter`. This helps in isolating relevant information.
3.  **Embedding**: Each chunk is converted into a numerical vector (embedding) using the `sentence-transformers/all-MiniLM-L6-v2` model. These embeddings capture the semantic meaning of the text.
4.  **Indexing**: The embeddings are stored in a FAISS index. FAISS allows for extremely fast similarity searches, enabling us to find the most relevant chunks for a given question instantly.
5.  **Retrieval & Answering**:
    -   When a user asks a question, it is also converted into an embedding.
    -   FAISS is used to search the index for the text chunks with embeddings most similar to the question's embedding.
    -   These top relevant chunks are concatenated to form a single `context`.
    -   The question and the context are passed to a `distilbert-base-uncased-distilled-squad` question-answering model, which extracts the final answer from the context.

---

## 🚀 Setup & Installation

1.  **Clone the repository or download the source code.**

2.  **Create a `docs` directory** in the root of the project and place your medical text files (`.txt`) inside it.
    ```bash
    mkdir docs
    # Now add your .txt files into this new 'docs' directory
    # For example: cp ~/path/to/my/medical_notes.txt docs/
    ```
 NB: If you want to add any further documents to the list, do create more .txt files in the docs folder, you can copy it from any website such as Wikipedia or download .txt files from the web and place it there in the docs folder.

3.  **Install the required Python libraries.** It is recommended to use a virtual environment.
    ```bash
    pip install streamlit sentence-transformers faiss-cpu transformers torch langchain-text-splitters numpy
    ```
    > **Note**: `faiss-cpu` is for systems without a dedicated NVIDIA GPU. If you have a CUDA-enabled GPU, you can install `faiss-gpu` for better performance: `pip install faiss-gpu`.

---

## ▶️ Usage

Once the setup is complete, you can run the Streamlit application from your terminal:

```bash
python rag_pipieline.py
```
