# Advanced Document Retrieval System

This project demonstrates an advanced document retrieval system using several key technologies, including `chromaDB` for vector database storage, `UMAP` for dimensionality reduction and visualization, `SentenceTransformers` for embeddings, and `Google Gemini AI` for query expansion and re-ranking of results. The project is built using Streamlit for the user interface.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Key Components](#key-components)
5. [Streamlit Interface](#streamlit-interface)
6. [Future Enhancements](#future-enhancements)
7. [License](#license)

## Features

- **PDF Reading & Chunking:** Reads PDF files, processes text into chunks using `RecursiveCharacterTextSplitter` and `SentenceTransformersTokenTextSplitter`.
- **Embedding and Storage:** Embeds text using `SentenceTransformer` and stores it in `chromaDB`.
- **UMAP Visualization:** Projects embeddings and queries into 2D space using UMAP for visualization.
- **Query Expansion:** Utilizes Google Gemini AI to generate hypothetical answers and multi-query expansion for more effective information retrieval.
- **Cross-Encoder Re-ranking:** Re-ranks results based on semantic similarity using `sentence-transformers CrossEncoder`.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/IbnuEyni/advanced-rag.git
    cd advanced-retrieval-system
    ```

2. **Install the required packages**  
    Use the `requirements.txt` file to install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install Streamlit**
    ```bash
    pip install streamlit
    ```

4. **Set up Google Gemini API key**  
    Add your `GEMINI_API_KEY` to a `secrets.toml` file in your Streamlit directory.

    ```toml
    [general]
    GEMINI_API_KEY = "your-google-gemini-api-key"
    ```

## Usage

To run the Streamlit app:

```bash
streamlit run app.py

```

## Demo

![Demo of Advanced Document Retrieval System](rag_gemini_strmlt.png)