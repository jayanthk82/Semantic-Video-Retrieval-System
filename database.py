import chromadb
import logging
import os
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('database')

class VectorDB:
    def __init__(self, collection_name='VIDEO_SEARCH'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.chroma_path = os.path.join(self.base_dir, "chroma_db")
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Connected to ChromaDB collection: {collection_name}")

    def insert_video(self, video_path, summary_text, vector_embedding):
        """Inserts a single video record into the database."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            video_name = os.path.basename(video_path)
            
            # Metadata dictionary
            meta = {
                'upload_time': timestamp,
                'path': video_path,
                'filename': video_name,
                'summary_snippet': summary_text[:100] + "..." # Store preview
            }

            self.collection.add(
                ids=[video_path],  # Using path as unique ID
                embeddings=[vector_embedding],
                documents=[summary_text],
                metadatas=[meta]
            )
            logger.info(f"Successfully inserted: {video_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert {video_path}: {e}")
            return False

    def search(self, query_embedding, n_results=3):
        """Queries the database for similar videos."""
        if self.collection.count() == 0:
            return None
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def get_all_files(self):
        """Retrieves metadata for all indexed files."""
        if self.collection.count() == 0:
            return []
        
        # ChromaDB .get() without args returns limited data, we just want metadata usually
        data = self.collection.get()
        return data.get('metadatas', [])

    def count(self):
        return self.collection.count()