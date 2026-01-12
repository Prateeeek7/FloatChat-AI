"""RAG system for natural language to SQL translation - optimized for 16GB RAM."""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import openai
from datetime import datetime, timedelta
import gc

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from database.models import DatabaseManager, VectorEmbedding, compress_array_data, decompress_array_data

logger = logging.getLogger(__name__)


class ARGORAGSystem:
    """Retrieval-Augmented Generation system for ARGO data queries."""
    
    def __init__(self, db_path: str = "data/floatchat.db", 
                 vector_db_path: str = "data/vector_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the RAG system with memory optimizations."""
        self.db_manager = DatabaseManager(db_path)
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model (lightweight for 16GB RAM)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.faiss_index = None
        self.vector_metadata = []
        
        # SQL query templates for common ARGO queries
        self.sql_templates = {
            'salinity_profiles': """
                SELECT p.platform_number, p.latitude, p.longitude, p.profile_date, 
                       p.salinity_data, p.pressure_data
                FROM argo_profiles p
                WHERE p.latitude BETWEEN :min_lat AND :max_lat
                AND p.longitude BETWEEN :min_lon AND :max_lon
                AND p.profile_date BETWEEN :start_date AND :end_date
                ORDER BY p.profile_date DESC
                LIMIT :limit
            """,
            'temperature_profiles': """
                SELECT p.platform_number, p.latitude, p.longitude, p.profile_date,
                       p.temperature_data, p.pressure_data
                FROM argo_profiles p
                WHERE p.latitude BETWEEN :min_lat AND :max_lat
                AND p.longitude BETWEEN :min_lon AND :max_lon
                AND p.profile_date BETWEEN :start_date AND :end_date
                ORDER BY p.profile_date DESC
                LIMIT :limit
            """,
            'trajectory_data': """
                SELECT t.platform_number, t.latitude, t.longitude, t.position_date
                FROM argo_trajectories t
                WHERE t.latitude BETWEEN :min_lat AND :max_lat
                AND t.longitude BETWEEN :min_lon AND :max_lon
                AND t.position_date BETWEEN :start_date AND :end_date
                ORDER BY t.position_date
            """,
            'platform_info': """
                SELECT f.platform_number, f.region, f.status, f.total_profiles,
                       f.first_deployment_date, f.last_profile_date
                FROM argo_floats f
                WHERE f.platform_number = :platform_number
            """,
            'nearest_floats': """
                SELECT f.platform_number, f.region, f.status,
                       MIN(ABS(t.latitude - :target_lat) + ABS(t.longitude - :target_lon)) as distance
                FROM argo_floats f
                JOIN argo_trajectories t ON f.platform_number = t.platform_number
                WHERE t.position_date >= :recent_date
                GROUP BY f.platform_number, f.region, f.status
                ORDER BY distance
                LIMIT :limit
            """
        }
        
        # Initialize vector database
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initialize or load FAISS vector database."""
        index_path = self.vector_db_path / "faiss_index.bin"
        metadata_path = self.vector_db_path / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            # Load existing index
            self.faiss_index = faiss.read_index(str(index_path))
            with open(metadata_path, 'r') as f:
                self.vector_metadata = json.load(f)
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
            self.vector_metadata = []
            logger.info("Created new FAISS index")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        # Process in batches to manage memory
        batch_size = 32  # Conservative for 16GB RAM
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            embeddings.append(batch_embeddings)
            
            # Memory cleanup
            if i % (batch_size * 4) == 0:
                gc.collect()
        
        return np.vstack(embeddings)
    
    def add_documents_to_vector_db(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector database."""
        texts = []
        metadata = []
        
        for doc in documents:
            # Create text representation
            text = self._create_document_text(doc)
            texts.append(text)
            metadata.append({
                'content_type': doc.get('content_type', 'unknown'),
                'content_id': doc.get('content_id', 0),
                'platform_number': doc.get('platform_number', 'unknown'),
                'summary': doc.get('summary', text[:200])
            })
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.generate_embeddings(texts)
        
        # Add to FAISS index
        self.faiss_index.add(embeddings.astype('float32'))
        self.vector_metadata.extend(metadata)
        
        # Save index and metadata
        self._save_vector_db()
        
        logger.info(f"Added {len(texts)} documents to vector database")
    
    def _create_document_text(self, doc: Dict[str, Any]) -> str:
        """Create text representation of a document for embedding."""
        content_type = doc.get('content_type', 'unknown')
        platform_number = doc.get('platform_number', 'unknown')
        
        if content_type == 'profile':
            return f"ARGO profile data from platform {platform_number}. " \
                   f"Location: {doc.get('latitude', 'unknown')}°N, {doc.get('longitude', 'unknown')}°E. " \
                   f"Date: {doc.get('profile_date', 'unknown')}. " \
                   f"Depth range: {doc.get('max_depth', 'unknown')}m. " \
                   f"Contains temperature and salinity measurements."
        
        elif content_type == 'trajectory':
            return f"ARGO trajectory data from platform {platform_number}. " \
                   f"Location: {doc.get('latitude', 'unknown')}°N, {doc.get('longitude', 'unknown')}°E. " \
                   f"Date: {doc.get('position_date', 'unknown')}. " \
                   f"Surface position data."
        
        elif content_type == 'metadata':
            return f"ARGO float metadata for platform {platform_number}. " \
                   f"Region: {doc.get('region', 'unknown')}. " \
                   f"Status: {doc.get('status', 'unknown')}. " \
                   f"Total profiles: {doc.get('total_profiles', 'unknown')}."
        
        else:
            return f"ARGO data from platform {platform_number}. " \
                   f"Type: {content_type}. " \
                   f"Location: {doc.get('latitude', 'unknown')}°N, {doc.get('longitude', 'unknown')}°E."
    
    def _save_vector_db(self):
        """Save FAISS index and metadata."""
        index_path = self.vector_db_path / "faiss_index.bin"
        metadata_path = self.vector_db_path / "metadata.json"
        
        faiss.write_index(self.faiss_index, str(index_path))
        with open(metadata_path, 'w') as f:
            json.dump(self.vector_metadata, f)
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if self.faiss_index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve relevant documents
        relevant_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.vector_metadata):
                doc_info = self.vector_metadata[idx].copy()
                doc_info['similarity_score'] = float(score)
                relevant_docs.append(doc_info)
        
        return relevant_docs
    
    def translate_query_to_sql(self, natural_language_query: str) -> Tuple[str, Dict[str, Any]]:
        """Translate natural language query to SQL using LLM."""
        # Retrieve relevant context
        relevant_docs = self.retrieve_relevant_documents(natural_language_query, top_k=3)
        
        # Create context from relevant documents
        context = self._create_context_from_docs(relevant_docs)
        
        # Create prompt for LLM
        prompt = self._create_sql_prompt(natural_language_query, context)
        
        try:
            # Use OpenAI API for SQL generation
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert SQL query generator for oceanographic ARGO float data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_response = response.choices[0].message.content.strip()
            
            # Parse SQL and parameters
            sql_query, parameters = self._parse_sql_response(sql_response)
            
            return sql_query, parameters
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            # Fallback to simple keyword matching
            return self._fallback_sql_generation(natural_language_query)
    
    def _create_context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Create context string from relevant documents."""
        context_parts = []
        
        for doc in docs:
            context_parts.append(f"- {doc.get('summary', 'No summary available')}")
        
        return "\n".join(context_parts)
    
    def _create_sql_prompt(self, query: str, context: str) -> str:
        """Create prompt for SQL generation."""
        return f"""
        Based on the following context about ARGO float data, generate a SQL query for this question:
        
        Question: {query}
        
        Context:
        {context}
        
        Available tables and columns:
        - argo_floats: platform_number, region, status, total_profiles, first_deployment_date, last_profile_date
        - argo_profiles: platform_number, cycle_number, julian_day, latitude, longitude, profile_date, 
                        pressure_data, temperature_data, salinity_data, max_depth
        - argo_trajectories: platform_number, julian_day, latitude, longitude, position_date, direction
        - argo_bgc_data: platform_number, cycle_number, variable_name, julian_day, latitude, longitude, values_data
        
        Important notes:
        - Use parameterized queries with :parameter_name format
        - For date ranges, use profile_date or position_date columns
        - For spatial queries, use latitude and longitude columns
        - Limit results to reasonable numbers (e.g., LIMIT 100)
        - Return JSON with 'sql' and 'parameters' fields
        
        Generate the SQL query:
        """
    
    def _parse_sql_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Parse SQL response from LLM."""
        try:
            # Try to parse as JSON first
            if response.startswith('{') and response.endswith('}'):
                parsed = json.loads(response)
                return parsed.get('sql', ''), parsed.get('parameters', {})
            
            # Otherwise, extract SQL from response
            lines = response.split('\n')
            sql_lines = []
            in_sql = False
            
            for line in lines:
                if line.strip().upper().startswith('SELECT'):
                    in_sql = True
                if in_sql:
                    sql_lines.append(line)
                if line.strip().endswith(';'):
                    break
            
            sql_query = '\n'.join(sql_lines).strip()
            return sql_query, {}
            
        except Exception as e:
            logger.error(f"Error parsing SQL response: {str(e)}")
            return response, {}
    
    def _fallback_sql_generation(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Fallback SQL generation using keyword matching."""
        query_lower = query.lower()
        
        if 'salinity' in query_lower:
            return self.sql_templates['salinity_profiles'], {
                'min_lat': -90, 'max_lat': 90,
                'min_lon': -180, 'max_lon': 180,
                'start_date': '2020-01-01', 'end_date': '2024-12-31',
                'limit': 100
            }
        elif 'temperature' in query_lower:
            return self.sql_templates['temperature_profiles'], {
                'min_lat': -90, 'max_lat': 90,
                'min_lon': -180, 'max_lon': 180,
                'start_date': '2020-01-01', 'end_date': '2024-12-31',
                'limit': 100
            }
        elif 'trajectory' in query_lower or 'path' in query_lower:
            return self.sql_templates['trajectory_data'], {
                'min_lat': -90, 'max_lat': 90,
                'min_lon': -180, 'max_lon': 180,
                'start_date': '2020-01-01', 'end_date': '2024-12-31'
            }
        else:
            return self.sql_templates['salinity_profiles'], {
                'min_lat': -90, 'max_lat': 90,
                'min_lon': -180, 'max_lon': 180,
                'start_date': '2020-01-01', 'end_date': '2024-12-31',
                'limit': 50
            }
    
    def execute_query(self, sql_query: str, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Execute SQL query and return results."""
        session = self.db_manager.get_session()
        
        try:
            result = session.execute(sql_query, parameters)
            columns = result.keys()
            rows = result.fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            # Decompress array data if present
            for col in df.columns:
                if col.endswith('_data') and df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: decompress_array_data(x) if x else [])
            
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query end-to-end."""
        logger.info(f"Processing query: {query}")
        
        # Translate to SQL
        sql_query, parameters = self.translate_query_to_sql(query)
        
        # Execute query
        results_df = self.execute_query(sql_query, parameters)
        
        # Generate response
        response = self._generate_response(query, results_df, sql_query)
        
        return {
            'query': query,
            'sql_query': sql_query,
            'parameters': parameters,
            'results': results_df.to_dict('records') if not results_df.empty else [],
            'response': response,
            'num_results': len(results_df)
        }
    
    def _generate_response(self, query: str, results_df: pd.DataFrame, sql_query: str) -> str:
        """Generate natural language response from query results."""
        if results_df.empty:
            return "No data found matching your query. Try adjusting your search criteria."
        
        num_results = len(results_df)
        
        if 'salinity' in query.lower():
            return f"Found {num_results} salinity profiles matching your query. " \
                   f"The data includes temperature and pressure measurements at various depths."
        elif 'temperature' in query.lower():
            return f"Found {num_results} temperature profiles matching your query. " \
                   f"The data includes salinity and pressure measurements at various depths."
        elif 'trajectory' in query.lower() or 'path' in query.lower():
            return f"Found {num_results} trajectory points matching your query. " \
                   f"These show the surface path of the ARGO floats."
        else:
            return f"Found {num_results} records matching your query. " \
                   f"Please review the data below for more details."


def main():
    """Example usage of the RAG system."""
    rag_system = ARGORAGSystem()
    
    # Example queries
    queries = [
        "Show me salinity profiles near the equator in March 2023",
        "Compare temperature data in the Arabian Sea for the last 6 months",
        "What are the nearest ARGO floats to this location?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag_system.process_natural_language_query(query)
        print(f"Response: {result['response']}")
        print(f"SQL: {result['sql_query']}")


if __name__ == "__main__":
    main()






