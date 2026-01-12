"""ARGO-specialized RAG system with domain-specific training."""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from datetime import datetime, timedelta
import gc

from ..database.models import DatabaseManager, VectorEmbedding, compress_array_data, decompress_array_data

logger = logging.getLogger(__name__)


class ARGOSpecializedRAG:
    """ARGO-specialized RAG system with domain knowledge."""
    
    def __init__(self, db_path: str = "data/floatchat.db", 
                 vector_db_path: str = "data/vector_db",
                 model_name: str = "microsoft/DialoGPT-small"):  # Lightweight for 16GB RAM
        """Initialize ARGO-specialized RAG system."""
        self.db_manager = DatabaseManager(db_path)
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Use a smaller, more efficient embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.faiss_index = None
        self.vector_metadata = []
        
        # ARGO-specific knowledge base
        self.argo_knowledge = self._build_argo_knowledge_base()
        
        # Initialize specialized model
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_specialized_model()
        
        # SQL templates with ARGO-specific context
        self.sql_templates = self._build_argo_sql_templates()
        
        # Initialize vector database
        self._initialize_vector_db()
    
    def _build_argo_knowledge_base(self) -> Dict[str, Any]:
        """Build ARGO-specific knowledge base."""
        return {
            "variables": {
                "TEMP": {
                    "name": "Temperature",
                    "unit": "°C",
                    "description": "Seawater temperature measured by ARGO floats",
                    "typical_range": "0-30",
                    "depth_coverage": "0-2000m"
                },
                "PSAL": {
                    "name": "Salinity",
                    "unit": "PSU (Practical Salinity Units)",
                    "description": "Seawater salinity measured by ARGO floats",
                    "typical_range": "30-40",
                    "depth_coverage": "0-2000m"
                },
                "PRES": {
                    "name": "Pressure",
                    "unit": "dbar",
                    "description": "Water pressure indicating depth",
                    "typical_range": "0-2000",
                    "depth_coverage": "Surface to 2000m"
                },
                "DOXY": {
                    "name": "Dissolved Oxygen",
                    "unit": "μmol/kg",
                    "description": "Oxygen concentration in seawater",
                    "typical_range": "0-400",
                    "depth_coverage": "0-2000m"
                }
            },
            "regions": {
                "Indian Ocean": {
                    "lat_range": (20, 40),
                    "lon_range": (60, 100),
                    "description": "Northern Indian Ocean region with ARGO floats",
                    "characteristics": "Seasonal monsoon, high salinity in Arabian Sea"
                },
                "Arabian Sea": {
                    "lat_range": (15, 25),
                    "lon_range": (60, 75),
                    "description": "Part of Indian Ocean with high salinity",
                    "characteristics": "High evaporation, low precipitation"
                },
                "Bay of Bengal": {
                    "lat_range": (10, 20),
                    "lon_range": (80, 95),
                    "description": "Northern Indian Ocean with low salinity",
                    "characteristics": "High freshwater input from rivers"
                }
            },
            "depth_zones": {
                "surface": (0, 10),
                "mixed_layer": (0, 50),
                "thermocline": (50, 200),
                "intermediate": (200, 1000),
                "deep": (1000, 2000)
            },
            "temporal_patterns": {
                "monsoon_seasons": {
                    "southwest_monsoon": "June-September",
                    "northeast_monsoon": "December-March",
                    "pre_monsoon": "March-May",
                    "post_monsoon": "October-November"
                }
            }
        }
    
    def _build_argo_sql_templates(self) -> Dict[str, str]:
        """Build ARGO-specific SQL templates with domain knowledge."""
        return {
            'salinity_profiles': """
                SELECT p.platform_number, p.latitude, p.longitude, p.profile_date, 
                       p.salinity_data, p.pressure_data, p.temperature_data
                FROM argo_profiles p
                WHERE p.latitude BETWEEN :min_lat AND :max_lat
                AND p.longitude BETWEEN :min_lon AND :max_lon
                AND p.profile_date BETWEEN :start_date AND :end_date
                AND p.salinity_data IS NOT NULL
                ORDER BY p.profile_date DESC
                LIMIT :limit
            """,
            'temperature_profiles': """
                SELECT p.platform_number, p.latitude, p.longitude, p.profile_date,
                       p.temperature_data, p.pressure_data, p.salinity_data
                FROM argo_profiles p
                WHERE p.latitude BETWEEN :min_lat AND :max_lat
                AND p.longitude BETWEEN :min_lon AND :max_lon
                AND p.profile_date BETWEEN :start_date AND :end_date
                AND p.temperature_data IS NOT NULL
                ORDER BY p.profile_date DESC
                LIMIT :limit
            """,
            'depth_specific_profiles': """
                SELECT p.platform_number, p.latitude, p.longitude, p.profile_date,
                       p.temperature_data, p.salinity_data, p.pressure_data
                FROM argo_profiles p
                WHERE p.latitude BETWEEN :min_lat AND :max_lat
                AND p.longitude BETWEEN :min_lon AND :max_lon
                AND p.profile_date BETWEEN :start_date AND :end_date
                AND p.max_depth >= :min_depth
                AND p.max_depth <= :max_depth
                ORDER BY p.profile_date DESC
                LIMIT :limit
            """,
            'monsoon_season_data': """
                SELECT p.platform_number, p.latitude, p.longitude, p.profile_date,
                       p.temperature_data, p.salinity_data, p.pressure_data
                FROM argo_profiles p
                WHERE p.latitude BETWEEN :min_lat AND :max_lat
                AND p.longitude BETWEEN :min_lon AND :max_lon
                AND strftime('%m', p.profile_date) IN :monsoon_months
                ORDER BY p.profile_date DESC
                LIMIT :limit
            """
        }
    
    def _load_specialized_model(self):
        """Load a specialized model for ARGO queries."""
        try:
            # Use a smaller model that fits in 16GB RAM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto"  # Automatically distribute across available memory
            )
            
            # Add ARGO-specific tokens
            argo_tokens = [
                "[ARGO]", "[TEMP]", "[PSAL]", "[PRES]", "[DOXY]",
                "[SALINITY]", "[TEMPERATURE]", "[PRESSURE]", "[OXYGEN]",
                "[INDIAN_OCEAN]", "[ARABIAN_SEA]", "[BAY_OF_BENGAL]",
                "[SURFACE]", "[THERMOCLINE]", "[DEEP]"
            ]
            
            self.tokenizer.add_tokens(argo_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Loaded specialized model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading specialized model: {str(e)}")
            # Fallback to simple template matching
            self.model = None
            self.tokenizer = None
    
    def _create_argo_context(self, query: str) -> str:
        """Create ARGO-specific context for the query."""
        context_parts = []
        
        # Add relevant variable information
        query_lower = query.lower()
        for var_key, var_info in self.argo_knowledge["variables"].items():
            if any(term in query_lower for term in [var_info["name"].lower(), var_key.lower()]):
                context_parts.append(
                    f"{var_info['name']} ({var_key}): {var_info['description']}, "
                    f"Unit: {var_info['unit']}, Range: {var_info['typical_range']}"
                )
        
        # Add regional information
        for region, region_info in self.argo_knowledge["regions"].items():
            if region.lower() in query_lower:
                context_parts.append(
                    f"{region}: {region_info['description']}, "
                    f"Latitude: {region_info['lat_range']}, Longitude: {region_info['lon_range']}"
                )
        
        # Add depth zone information
        if any(term in query_lower for term in ["depth", "surface", "deep", "thermocline"]):
            context_parts.append("Depth zones: Surface (0-10m), Mixed layer (0-50m), "
                               "Thermocline (50-200m), Intermediate (200-1000m), Deep (1000-2000m)")
        
        return " | ".join(context_parts)
    
    def _initialize_vector_db(self):
        """Initialize or load FAISS vector database."""
        index_path = self.vector_db_path / "argo_faiss_index.bin"
        metadata_path = self.vector_db_path / "argo_metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            self.faiss_index = faiss.read_index(str(index_path))
            with open(metadata_path, 'r') as f:
                self.vector_metadata = json.load(f)
            logger.info(f"Loaded ARGO FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.vector_metadata = []
            logger.info("Created new ARGO FAISS index")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for ARGO-specific texts."""
        batch_size = 16  # Smaller batch for memory efficiency
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            embeddings.append(batch_embeddings)
            
            if i % (batch_size * 4) == 0:
                gc.collect()
        
        return np.vstack(embeddings)
    
    def translate_query_to_sql(self, natural_language_query: str) -> Tuple[str, Dict[str, Any]]:
        """Translate natural language query to SQL using ARGO-specific knowledge."""
        # Create ARGO-specific context
        argo_context = self._create_argo_context(natural_language_query)
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(natural_language_query, top_k=3)
        
        # Combine context
        full_context = f"ARGO Context: {argo_context}\nRelevant Data: {self._create_context_from_docs(relevant_docs)}"
        
        # Use specialized model or fallback
        if self.model and self.tokenizer:
            return self._generate_sql_with_specialized_model(natural_language_query, full_context)
        else:
            return self._generate_sql_with_templates(natural_language_query, full_context)
    
    def _generate_sql_with_specialized_model(self, query: str, context: str) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL using the specialized ARGO model."""
        try:
            # Create prompt for ARGO-specific model
            prompt = f"""
            [ARGO] Generate SQL query for oceanographic data analysis.
            
            Query: {query}
            Context: {context}
            
            Available tables:
            - argo_floats: platform_number, region, status, total_profiles
            - argo_profiles: platform_number, cycle_number, latitude, longitude, 
                           profile_date, temperature_data, salinity_data, pressure_data
            - argo_trajectories: platform_number, latitude, longitude, position_date
            
            Generate SQL query with parameters:
            """
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract SQL from response
            sql_query, parameters = self._parse_argo_sql_response(response)
            
            return sql_query, parameters
            
        except Exception as e:
            logger.error(f"Error with specialized model: {str(e)}")
            return self._generate_sql_with_templates(query, context)
    
    def _generate_sql_with_templates(self, query: str, context: str) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL using ARGO-specific templates."""
        query_lower = query.lower()
        
        # ARGO-specific keyword matching
        if "salinity" in query_lower or "salt" in query_lower:
            template = self.sql_templates['salinity_profiles']
            parameters = self._extract_argo_parameters(query, "salinity")
            
        elif "temperature" in query_lower or "temp" in query_lower:
            template = self.sql_templates['temperature_profiles']
            parameters = self._extract_argo_parameters(query, "temperature")
            
        elif any(term in query_lower for term in ["depth", "surface", "deep", "thermocline"]):
            template = self.sql_templates['depth_specific_profiles']
            parameters = self._extract_argo_parameters(query, "depth")
            
        elif any(term in query_lower for term in ["monsoon", "season", "seasonal"]):
            template = self.sql_templates['monsoon_season_data']
            parameters = self._extract_argo_parameters(query, "monsoon")
            
        else:
            template = self.sql_templates['salinity_profiles']
            parameters = self._extract_argo_parameters(query, "general")
        
        return template, parameters
    
    def _extract_argo_parameters(self, query: str, query_type: str) -> Dict[str, Any]:
        """Extract parameters with ARGO-specific knowledge."""
        parameters = {
            'min_lat': -90, 'max_lat': 90,
            'min_lon': -180, 'max_lon': 180,
            'start_date': '2020-01-01', 'end_date': '2024-12-31',
            'limit': 100
        }
        
        query_lower = query.lower()
        
        # Extract regional information
        if "indian ocean" in query_lower:
            parameters.update({'min_lat': 20, 'max_lat': 40, 'min_lon': 60, 'max_lon': 100})
        elif "arabian sea" in query_lower:
            parameters.update({'min_lat': 15, 'max_lat': 25, 'min_lon': 60, 'max_lon': 75})
        elif "bay of bengal" in query_lower:
            parameters.update({'min_lat': 10, 'max_lat': 20, 'min_lon': 80, 'max_lon': 95})
        elif "equator" in query_lower:
            parameters.update({'min_lat': -5, 'max_lat': 5})
        
        # Extract depth information
        if "surface" in query_lower:
            parameters.update({'min_depth': 0, 'max_depth': 50})
        elif "deep" in query_lower:
            parameters.update({'min_depth': 1000, 'max_depth': 2000})
        elif "thermocline" in query_lower:
            parameters.update({'min_depth': 50, 'max_depth': 200})
        
        # Extract temporal information
        if "march" in query_lower:
            parameters.update({'start_date': '2023-03-01', 'end_date': '2023-03-31'})
        elif "last 6 months" in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            parameters.update({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
        
        return parameters
    
    def _parse_argo_sql_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Parse SQL response with ARGO-specific handling."""
        try:
            # Extract SQL query from response
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
            
            # Extract parameters (simplified for demo)
            parameters = self._extract_argo_parameters(sql_query, "general")
            
            return sql_query, parameters
            
        except Exception as e:
            logger.error(f"Error parsing ARGO SQL response: {str(e)}")
            return response, {}
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant ARGO documents."""
        if self.faiss_index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.vector_metadata):
                doc_info = self.vector_metadata[idx].copy()
                doc_info['similarity_score'] = float(score)
                relevant_docs.append(doc_info)
        
        return relevant_docs
    
    def _create_context_from_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Create context from relevant documents."""
        context_parts = []
        for doc in docs:
            context_parts.append(f"- {doc.get('summary', 'No summary available')}")
        return "\n".join(context_parts)
    
    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query with ARGO specialization."""
        logger.info(f"Processing ARGO query: {query}")
        
        # Translate to SQL
        sql_query, parameters = self.translate_query_to_sql(query)
        
        # Execute query
        results_df = self.execute_query(sql_query, parameters)
        
        # Generate ARGO-specific response
        response = self._generate_argo_response(query, results_df, sql_query)
        
        return {
            'query': query,
            'sql_query': sql_query,
            'parameters': parameters,
            'results': results_df.to_dict('records') if not results_df.empty else [],
            'response': response,
            'num_results': len(results_df)
        }
    
    def _generate_argo_response(self, query: str, results_df: pd.DataFrame, sql_query: str) -> str:
        """Generate ARGO-specific response."""
        if results_df.empty:
            return "No ARGO data found matching your query. Try adjusting your search criteria or check if the region has active floats."
        
        num_results = len(results_df)
        query_lower = query.lower()
        
        # ARGO-specific responses
        if "salinity" in query_lower:
            return f"Found {num_results} salinity profiles from ARGO floats. " \
                   f"Salinity data shows seawater salt content in PSU (Practical Salinity Units), " \
                   f"typically ranging from 30-40 PSU in ocean waters."
        elif "temperature" in query_lower:
            return f"Found {num_results} temperature profiles from ARGO floats. " \
                   f"Temperature data shows seawater temperature in °C, " \
                   f"ranging from near-freezing in deep waters to 30°C+ in tropical surface waters."
        elif "trajectory" in query_lower or "path" in query_lower:
            return f"Found {num_results} ARGO float trajectory points. " \
                   f"These show the surface drift path of autonomous profiling floats " \
                   f"as they move with ocean currents."
        else:
            return f"Found {num_results} ARGO float records matching your query. " \
                   f"ARGO floats provide continuous oceanographic measurements including " \
                   f"temperature, salinity, and pressure profiles from the surface to 2000m depth."
    
    def execute_query(self, sql_query: str, parameters: Dict[str, Any]) -> pd.DataFrame:
        """Execute SQL query and return results."""
        session = self.db_manager.get_session()
        
        try:
            result = session.execute(sql_query, parameters)
            columns = result.keys()
            rows = result.fetchall()
            
            df = pd.DataFrame(rows, columns=columns)
            
            # Decompress array data if present
            for col in df.columns:
                if col.endswith('_data') and df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: decompress_array_data(x) if x else [])
            
            return df
            
        except Exception as e:
            logger.error(f"Error executing ARGO query: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()


def main():
    """Test the ARGO-specialized RAG system."""
    rag_system = ARGOSpecializedRAG()
    
    # Test ARGO-specific queries
    test_queries = [
        "Show me salinity profiles in the Arabian Sea during monsoon season",
        "Find temperature data in the thermocline layer of the Indian Ocean",
        "What are the ARGO float trajectories in the Bay of Bengal?",
        "Show me deep ocean pressure profiles from ARGO floats"
    ]
    
    for query in test_queries:
        print(f"\nARGO Query: {query}")
        result = rag_system.process_natural_language_query(query)
        print(f"Response: {result['response']}")
        print(f"SQL: {result['sql_query']}")


if __name__ == "__main__":
    main()
