import tempfile
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import umap


from utils import *
from utils import _read_pdf
# Streamlit UI
st.title("PDF Document Query and Ranking")

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

embedding_function = SentenceTransformerEmbeddingFunction()
# Load Chroma collection
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    # Create index
    st.write("Creating index...")
    collection_name = create_index_pdf(temp_file_path)
    st.write(f"Index created with collection name: {collection_name}")
    # Load the Chroma collection using the read PDF text
    chroma_collection = load_chroma(temp_file_path, collection_name=collection_name, embedding_function=embedding_function)
    st.write(f"Loaded {chroma_collection.count()} chunks.")

# if uploaded_file is not None:
#     st.write("Loading PDF and creating Chroma collection...")
#     st.write(f"Uploaded file: {uploaded_file.name}")

#     # Read the uploaded file
#     pdf_texts = _read_pdf(uploaded_file)

#     # Load the Chroma collection using the read PDF text
#     chroma_collection = load_chroma(pdf_texts, collection_name=uploaded_file.name, embedding_function=embedding_function)
#     st.write(f"Loaded {chroma_collection.count()} chunks.")

query = st.text_input('Enter your query:')

if query:

    results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])
    
    if results['documents']:
        retrieved_documents = results['documents'][0]

        # Display retrieved documents
        for document in retrieved_documents:
            st.write(word_wrap(document))
        
        # Embedding projection for query
        embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
        umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
        
        projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
        query_embedding = embedding_function.embed([query])[0]
        retrieved_embeddings = results['embeddings'][0]
        
        projected_query_embedding = project_embeddings([query_embedding], umap_transform)
        projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

        # Generate expanded query and perform a new retrieval
        joint_query = expand_query_with_answer(query)
        print(joint_query)
        
        results = chroma_collection.query(query_texts=joint_query, n_results=5, include=['documents', 'embeddings'])
        retrieved_documents = results['documents'][0]

        for doc in retrieved_documents:
            print(word_wrap(doc))
            print('')

        # Multi-query expansion to cover different aspects
        augmented_queries = augment_multiple_query(query)
        queries = [query] + augmented_queries

        # Perform multi-query retrieval
        results = chroma_collection.query(query_texts=queries, n_results=5, include=['documents', 'embeddings'])
        retrieved_documents = results['documents']

        # Deduplicate and display results
        unique_documents = set()
        for documents in retrieved_documents:
            for document in documents:
                unique_documents.add(document)

        for i, documents in enumerate(retrieved_documents):
            print(f"Query: {queries[i]}")
            print('')
            print("Results:")
            for doc in documents:
                print(word_wrap(doc))
                print('')
            print('-'*100)

        # Re-ranking documents using CrossEncoder
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        ranked_docs = rank_documents(cross_encoder, query, retrieved_documents[0])

        # Display ranked documents
        for rank, doc in ranked_docs.items():
            print(f"Rank {rank + 1}:")
            print(word_wrap(doc))
            print('')
