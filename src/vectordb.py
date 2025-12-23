import os
import logging
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Configure logging for VectorDB
logger = logging.getLogger(__name__)


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")
        logger.info(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Use LangChain's RecursiveCharacterTextSplitter for better chunking
        # It handles sentence boundaries and preserves context better
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,  # Overlap between chunks to preserve context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents (each should be a dict with 'content' and 'metadata' keys)
        """
        print(f"Processing {len(documents)} documents...")
        logger.info(f"Processing {len(documents)} documents for ingestion")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            if isinstance(doc, dict):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                source = metadata.get("source", f"doc_{doc_idx}")
            else:
                # Handle case where document is just a string
                content = str(doc)
                metadata = {}
                source = f"doc_{doc_idx}"
            
            if not content:
                logger.warning(f"Skipping empty document at index {doc_idx}")
                continue
            
            # Split document into chunks
            chunks = self.chunk_text(content)
            logger.debug(f"Document {source} split into {len(chunks)} chunks")
            
            # Prepare data for ChromaDB
            if chunks:
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                    all_chunks.append(chunk)
                    all_metadatas.append({**metadata, "chunk_index": chunk_idx, "doc_index": doc_idx})
                    all_ids.append(chunk_id)
        
        # Add all chunks to ChromaDB collection
        if all_chunks:
            logger.info(f"Creating embeddings for {len(all_chunks)} chunks...")
            # Create embeddings for all chunks at once (more efficient)
            embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=False)
            
            # Convert embeddings to list format for ChromaDB
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
            logger.info(f"Adding {len(all_chunks)} chunks to ChromaDB collection...")
            self.collection.add(
                ids=all_ids,
                embeddings=embeddings_list,
                documents=all_chunks,
                metadatas=all_metadatas
            )
            print(f"Added {len(all_chunks)} chunks from {len(documents)} documents to vector database")
            logger.info(f"Successfully added {len(all_chunks)} chunks from {len(documents)} documents to vector database")
        else:
            print("No chunks to add to vector database")
            logger.warning("No chunks to add to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        logger.debug(f"Creating query embedding for: {query[:100]}...")
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        
        # Convert to list format for ChromaDB
        query_embedding_list = query_embedding[0].tolist() if hasattr(query_embedding[0], 'tolist') else query_embedding[0]
        
        # Search in ChromaDB collection
        logger.debug(f"Searching ChromaDB for top {n_results} results...")
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results
        )
        
        # Format results to match expected structure
        formatted_results = {
            "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
            "metadatas": results.get("metadatas", [[]])[0] if results.get("metadatas") else [],
            "distances": results.get("distances", [[]])[0] if results.get("distances") else [],
            "ids": results.get("ids", [[]])[0] if results.get("ids") else [],
        }
        
        num_results = len(formatted_results["documents"])
        if formatted_results.get("distances"):
            avg_distance = sum(formatted_results["distances"]) / len(formatted_results["distances"])
            logger.info(f"Retrieved {num_results} results (avg distance: {avg_distance:.4f})")
        else:
            logger.info(f"Retrieved {num_results} results")
        
        return formatted_results