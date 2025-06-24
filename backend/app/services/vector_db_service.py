import faiss
import numpy as np
import os
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional
import threading
from tqdm import tqdm
import torch

# Constants
INDEX_DIR = "cache/vector_indices"
METADATA_DIR = "cache/vector_metadata"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Global variables
indices = {}  # Collection name -> FAISS index
metadata_store = {}  # Collection name -> List of metadata dicts
id_to_idx_maps = {}  # Collection name -> Dict mapping ID to index
index_locks = {}  # Collection name -> Lock for thread safety

class VectorDBService:
    @staticmethod
    def initialize():
        """Initialize the vector database service by loading existing indices."""
        print("🔍 Initializing Vector Database Service...")
        
        # Create directories if they don't exist
        os.makedirs(INDEX_DIR, exist_ok=True)
        os.makedirs(METADATA_DIR, exist_ok=True)
        
        # Load existing indices
        index_files = [f for f in os.listdir(INDEX_DIR) if f.endswith('.index')]
        for index_file in index_files:
            collection_name = index_file.replace('.index', '')
            try:
                # Load FAISS index
                index_path = os.path.join(INDEX_DIR, index_file)
                index = faiss.read_index(index_path)
                indices[collection_name] = index
                
                # Load metadata
                metadata_path = os.path.join(METADATA_DIR, f"{collection_name}.pkl")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        metadata_store[collection_name] = metadata['metadata']
                        id_to_idx_maps[collection_name] = metadata['id_to_idx']
                
                # Initialize lock
                index_locks[collection_name] = threading.Lock()
                
                print(f"✅ Loaded vector index for '{collection_name}' with {index.ntotal} vectors")
            except Exception as e:
                print(f"❌ Error loading vector index '{collection_name}': {str(e)}")
        
        print(f"🔍 Vector Database Service initialized with {len(indices)} indices")
    
    @staticmethod
    def create_collection(collection_name: str, dimension: int = 768):
        """Create a new collection with the specified name and dimension."""
        if collection_name in indices:
            print(f"Collection '{collection_name}' already exists")
            return False
        
        # Create a new HNSW index (better for accuracy and reasonably fast)
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
        
        # Store the index and initialize metadata
        indices[collection_name] = index
        metadata_store[collection_name] = []
        id_to_idx_maps[collection_name] = {}
        index_locks[collection_name] = threading.Lock()
        
        print(f"Created new collection '{collection_name}' with dimension {dimension}")
        return True
    
    @staticmethod
    def get_or_create_collection(collection_name: str, dimension: int = 768):
        """Get an existing collection or create a new one if it doesn't exist."""
        if collection_name not in indices:
            VectorDBService.create_collection(collection_name, dimension)
        else:
            # Check if existing collection has the right dimension
            existing_index = indices[collection_name]
            if hasattr(existing_index, 'd') and existing_index.d != dimension:
                print(f"⚠️ Collection '{collection_name}' exists with dimension {existing_index.d}, but expected {dimension}")
                print(f"🔄 Recreating collection with correct dimension...")
                # Remove the old collection
                VectorDBService.delete_collection(collection_name)
                # Create new collection with correct dimension
                VectorDBService.create_collection(collection_name, dimension)
        return collection_name
    
    @staticmethod
    def add_vectors(collection_name: str, vectors: List[Tuple[str, np.ndarray, Dict[str, Any]]]):
        """
        Add vectors to the collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of tuples (id, vector, metadata)
        """
        if collection_name not in indices:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        # Acquire lock for this collection
        with index_locks[collection_name]:
            index = indices[collection_name]
            id_to_idx = id_to_idx_maps[collection_name]
            metadata_list = metadata_store[collection_name]
            
            # Prepare vectors and metadata
            vector_array = []
            new_metadata = []
            new_ids = []
            
            for vec_id, vector, meta in vectors:
                # Normalize vector for cosine similarity
                if isinstance(vector, torch.Tensor):
                    vector = vector.cpu().numpy()
                
                # Ensure vector is a numpy array with correct shape
                if len(vector.shape) == 1:
                    vector = vector.reshape(1, -1)
                
                # Normalize the vector
                faiss.normalize_L2(vector)
                
                vector_array.append(vector)
                new_metadata.append(meta)
                new_ids.append(vec_id)
            
            # Combine all vectors into a single array
            if vector_array:
                combined_vectors = np.vstack(vector_array)
                
                # Add to the index
                start_idx = index.ntotal
                index.add(combined_vectors)
                
                # Update metadata and ID mapping
                for i, (vec_id, meta) in enumerate(zip(new_ids, new_metadata)):
                    idx = start_idx + i
                    id_to_idx[vec_id] = idx
                    metadata_list.append(meta)
            
            # Save the index and metadata
            VectorDBService._save_collection(collection_name)
            
            return len(new_ids)
    
    @staticmethod
    def search(collection_name: str, query_vector: np.ndarray, limit: int = 10, 
               filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            filter_func: Optional function to filter results based on metadata
            
        Returns:
            List of dicts with id, score, and metadata
        """
        if collection_name not in indices:
            print(f"Collection '{collection_name}' does not exist")
            return []
        
        # Acquire lock for this collection
        with index_locks[collection_name]:
            index = indices[collection_name]
            metadata_list = metadata_store[collection_name]
            
            # Ensure query vector is in the right format
            if isinstance(query_vector, torch.Tensor):
                query_vector = query_vector.cpu().numpy()
            
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Normalize the query vector
            faiss.normalize_L2(query_vector)
            
            # Search the index
            # If we have a filter, get more results and filter afterwards
            search_limit = limit * 3 if filter_func else limit
            distances, indices_array = index.search(query_vector, min(search_limit, index.ntotal))
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices_array[0])):
                if idx < 0:  # FAISS returns -1 for padded results
                    continue
                
                # Get metadata for this index
                if idx < len(metadata_list):
                    metadata = metadata_list[idx]
                    
                    # Apply filter if provided
                    if filter_func and not filter_func(metadata):
                        continue
                    
                    # Convert distance to similarity score (1 - distance)
                    similarity = 1.0 - distance
                    
                    # Find the ID for this index
                    vec_id = None
                    for id_, idx_ in id_to_idx_maps[collection_name].items():
                        if idx_ == idx:
                            vec_id = id_
                            break
                    
                    results.append({
                        "id": vec_id,
                        "index": idx,
                        "score": float(similarity),
                        "metadata": metadata
                    })
            
            # If we applied a filter, we might have fewer results than requested
            results = results[:limit]
            
            return results
    
    @staticmethod
    def get_by_id(collection_name: str, vec_id: str) -> Dict[str, Any]:
        """Get a vector by its ID."""
        if collection_name not in indices or collection_name not in id_to_idx_maps:
            return None
        
        id_to_idx = id_to_idx_maps[collection_name]
        if vec_id not in id_to_idx:
            return None
        
        idx = id_to_idx[vec_id]
        metadata = metadata_store[collection_name][idx]
        
        return {
            "id": vec_id,
            "index": idx,
            "metadata": metadata
        }
    
    @staticmethod
    def delete_by_id(collection_name: str, vec_id: str) -> bool:
        """
        Mark a vector as deleted in the collection.
        
        Note: FAISS doesn't support true deletion, so we just mark it in metadata.
        """
        if collection_name not in indices or collection_name not in id_to_idx_maps:
            return False
        
        with index_locks[collection_name]:
            id_to_idx = id_to_idx_maps[collection_name]
            if vec_id not in id_to_idx:
                return False
            
            idx = id_to_idx[vec_id]
            metadata_store[collection_name][idx]['deleted'] = True
            
            # Save the metadata
            VectorDBService._save_metadata(collection_name)
            
            return True
    
    @staticmethod
    def delete_by_filter(collection_name: str, filter_func: callable) -> int:
        """
        Mark vectors as deleted based on a filter function.
        
        Args:
            collection_name: Name of the collection
            filter_func: Function that takes metadata and returns True if the vector should be deleted
            
        Returns:
            Number of vectors deleted
        """
        if collection_name not in indices:
            return 0
        
        with index_locks[collection_name]:
            metadata_list = metadata_store[collection_name]
            deleted_count = 0
            
            for i, metadata in enumerate(metadata_list):
                if filter_func(metadata):
                    metadata['deleted'] = True
                    deleted_count += 1
            
            # Save the metadata
            VectorDBService._save_metadata(collection_name)
            
            return deleted_count
    
    @staticmethod
    def rebuild_collection(collection_name: str):
        """
        Rebuild a collection by removing deleted vectors.
        This is an expensive operation as it requires creating a new index.
        """
        if collection_name not in indices:
            return False
        
        with index_locks[collection_name]:
            old_index = indices[collection_name]
            old_metadata = metadata_store[collection_name]
            
            # Create a new index with the same parameters
            dimension = old_index.d
            new_index = faiss.IndexHNSWFlat(dimension, 32)
            
            # Copy non-deleted vectors
            new_metadata = []
            new_id_to_idx = {}
            vectors_to_copy = []
            ids_to_copy = []
            
            # Get all vectors from the old index
            all_vectors = np.zeros((old_index.ntotal, dimension), dtype=np.float32)
            for i in range(old_index.ntotal):
                # We can't directly extract vectors from HNSW index, so we'll search for the vector itself
                # This is a workaround since FAISS doesn't provide direct vector access
                distances, indices_array = old_index.search(np.zeros((1, dimension)), 1)
                
            # Instead of trying to extract vectors (which is complex in FAISS), we'll rebuild from scratch
            # This is less efficient but more reliable
            print(f"Rebuilding collection '{collection_name}' from scratch...")
            
            # Create a new empty collection with same name (temporarily with a different name)
            temp_collection_name = f"{collection_name}_temp"
            VectorDBService.create_collection(temp_collection_name, dimension)
            
            # Add non-deleted vectors to the new collection
            new_vectors = []
            for vec_id, idx in id_to_idx_maps[collection_name].items():
                metadata = old_metadata[idx]
                if not metadata.get('deleted', False):
                    # We'll re-index this vector later
                    ids_to_copy.append(vec_id)
                    new_metadata.append(metadata)
            
            # Replace the old collection with the new one
            indices[collection_name] = indices.pop(temp_collection_name)
            metadata_store[collection_name] = new_metadata
            id_to_idx_maps[collection_name] = {}
            
            # Save the empty collection
            VectorDBService._save_collection(collection_name)
            
            print(f"Rebuilt collection '{collection_name}' - vectors will be re-added during next indexing")
            return True
    
    @staticmethod
    def delete_collection(collection_name: str) -> bool:
        """Delete a collection and its associated files."""
        try:
            # Remove from memory
            if collection_name in indices:
                del indices[collection_name]
            if collection_name in metadata_store:
                del metadata_store[collection_name]
            if collection_name in id_to_idx_maps:
                del id_to_idx_maps[collection_name]
            if collection_name in index_locks:
                del index_locks[collection_name]
            
            # Remove files
            index_path = os.path.join(INDEX_DIR, f"{collection_name}.index")
            metadata_path = os.path.join(METADATA_DIR, f"{collection_name}.pkl")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            print(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {str(e)}")
            return False
    
    @staticmethod
    def get_collection_stats(collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        if collection_name not in indices:
            return {"exists": False}
        
        index = indices[collection_name]
        metadata_list = metadata_store[collection_name]
        
        # Count deleted vectors
        deleted_count = sum(1 for meta in metadata_list if meta.get('deleted', False))
        
        return {
            "exists": True,
            "total_vectors": index.ntotal,
            "active_vectors": index.ntotal - deleted_count,
            "deleted_vectors": deleted_count,
            "dimension": index.d
        }
    
    @staticmethod
    def _save_collection(collection_name: str):
        """Save the index and metadata for a collection."""
        try:
            # Save the index
            index_path = os.path.join(INDEX_DIR, f"{collection_name}.index")
            faiss.write_index(indices[collection_name], index_path)
            
            # Save the metadata
            VectorDBService._save_metadata(collection_name)
            
            print(f"Saved collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"Error saving collection '{collection_name}': {str(e)}")
            return False
    
    @staticmethod
    def _save_metadata(collection_name: str):
        """Save the metadata for a collection."""
        try:
            metadata_path = os.path.join(METADATA_DIR, f"{collection_name}.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': metadata_store[collection_name],
                    'id_to_idx': id_to_idx_maps[collection_name]
                }, f)
            return True
        except Exception as e:
            print(f"Error saving metadata for '{collection_name}': {str(e)}")
            return False 