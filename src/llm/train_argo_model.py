"""Training script for ARGO-specialized language model."""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class ARGOModelTrainer:
    """Trainer for ARGO-specialized language model."""
    
    def __init__(self, base_model: str = "microsoft/DialoGPT-small"):
        """Initialize the trainer with a lightweight base model."""
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        
    def create_argo_training_data(self) -> List[Dict[str, str]]:
        """Create ARGO-specific training data."""
        training_examples = [
            # Salinity queries
            {
                "input": "Show me salinity profiles in the Indian Ocean",
                "output": "SELECT platform_number, latitude, longitude, profile_date, salinity_data, pressure_data FROM argo_profiles WHERE latitude BETWEEN 20 AND 40 AND longitude BETWEEN 60 AND 100 AND salinity_data IS NOT NULL ORDER BY profile_date DESC LIMIT 100"
            },
            {
                "input": "Find salinity data in the Arabian Sea during monsoon",
                "output": "SELECT platform_number, latitude, longitude, profile_date, salinity_data FROM argo_profiles WHERE latitude BETWEEN 15 AND 25 AND longitude BETWEEN 60 AND 75 AND strftime('%m', profile_date) IN ('06', '07', '08', '09') ORDER BY profile_date DESC LIMIT 100"
            },
            {
                "input": "Show me surface salinity near the equator",
                "output": "SELECT platform_number, latitude, longitude, profile_date, salinity_data FROM argo_profiles WHERE latitude BETWEEN -5 AND 5 AND salinity_data IS NOT NULL ORDER BY profile_date DESC LIMIT 100"
            },
            
            # Temperature queries
            {
                "input": "Find temperature profiles in the Bay of Bengal",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, pressure_data FROM argo_profiles WHERE latitude BETWEEN 10 AND 20 AND longitude BETWEEN 80 AND 95 AND temperature_data IS NOT NULL ORDER BY profile_date DESC LIMIT 100"
            },
            {
                "input": "Show me deep ocean temperature data",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, pressure_data FROM argo_profiles WHERE max_depth >= 1000 AND temperature_data IS NOT NULL ORDER BY profile_date DESC LIMIT 100"
            },
            {
                "input": "Find thermocline temperature profiles",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, pressure_data FROM argo_profiles WHERE max_depth BETWEEN 50 AND 200 AND temperature_data IS NOT NULL ORDER BY profile_date DESC LIMIT 100"
            },
            
            # Trajectory queries
            {
                "input": "Show me ARGO float trajectories in the Indian Ocean",
                "output": "SELECT platform_number, latitude, longitude, position_date FROM argo_trajectories WHERE latitude BETWEEN 20 AND 40 AND longitude BETWEEN 60 AND 100 ORDER BY position_date"
            },
            {
                "input": "Find float paths near coordinates 20N 70E",
                "output": "SELECT platform_number, latitude, longitude, position_date FROM argo_trajectories WHERE ABS(latitude - 20) < 5 AND ABS(longitude - 70) < 5 ORDER BY position_date"
            },
            
            # Platform-specific queries
            {
                "input": "Show me data from platform 2901503",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, salinity_data FROM argo_profiles WHERE platform_number = '2901503' ORDER BY profile_date DESC LIMIT 100"
            },
            {
                "input": "Find all profiles from active ARGO floats",
                "output": "SELECT p.platform_number, p.latitude, p.longitude, p.profile_date, p.temperature_data, p.salinity_data FROM argo_profiles p JOIN argo_floats f ON p.platform_number = f.platform_number WHERE f.status = 'active' ORDER BY p.profile_date DESC LIMIT 100"
            },
            
            # Temporal queries
            {
                "input": "Show me data from March 2023",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, salinity_data FROM argo_profiles WHERE profile_date BETWEEN '2023-03-01' AND '2023-03-31' ORDER BY profile_date DESC LIMIT 100"
            },
            {
                "input": "Find recent data from last 6 months",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, salinity_data FROM argo_profiles WHERE profile_date >= date('now', '-6 months') ORDER BY profile_date DESC LIMIT 100"
            },
            
            # Combined queries
            {
                "input": "Show me salinity and temperature profiles in the Arabian Sea during southwest monsoon",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, salinity_data, pressure_data FROM argo_profiles WHERE latitude BETWEEN 15 AND 25 AND longitude BETWEEN 60 AND 75 AND strftime('%m', profile_date) IN ('06', '07', '08', '09') AND temperature_data IS NOT NULL AND salinity_data IS NOT NULL ORDER BY profile_date DESC LIMIT 100"
            },
            {
                "input": "Find deep ocean data with both temperature and salinity",
                "output": "SELECT platform_number, latitude, longitude, profile_date, temperature_data, salinity_data, pressure_data FROM argo_profiles WHERE max_depth >= 1000 AND temperature_data IS NOT NULL AND salinity_data IS NOT NULL ORDER BY profile_date DESC LIMIT 100"
            }
        ]
        
        return training_examples
    
    def prepare_dataset(self) -> Dataset:
        """Prepare the training dataset."""
        training_data = self.create_argo_training_data()
        
        # Convert to the format expected by the model
        texts = []
        for example in training_data:
            # Create a conversation format
            text = f"Human: {example['input']}\nAssistant: {example['output']}<|endoftext|>"
            texts.append(text)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize the examples."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def train_model(self, output_dir: str = "models/argo_specialized"):
        """Train the ARGO-specialized model."""
        logger.info("Starting ARGO model training...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add ARGO-specific tokens
        argo_tokens = [
            "[ARGO]", "[TEMP]", "[PSAL]", "[PRES]", "[DOXY]",
            "[SALINITY]", "[TEMPERATURE]", "[PRESSURE]", "[OXYGEN]",
            "[INDIAN_OCEAN]", "[ARABIAN_SEA]", "[BAY_OF_BENGAL]",
            "[SURFACE]", "[THERMOCLINE]", "[DEEP]", "[MONSOON]"
        ]
        
        self.tokenizer.add_tokens(argo_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Low rank for memory efficiency
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]  # Only target attention layers
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Training arguments optimized for 16GB RAM
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Small batch size
            gradient_accumulation_steps=4,  # Accumulate gradients
            num_train_epochs=3,
            learning_rate=5e-5,
            fp16=True,  # Use half precision
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Save memory
            gradient_checkpointing=True,  # Save memory
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="steps",
            warmup_steps=50,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        logger.info("Training started...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
        return output_dir
    
    def evaluate_model(self, test_queries: List[str]) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained yet")
        
        results = []
        
        for query in test_queries:
            # Create prompt
            prompt = f"Human: {query}\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
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
            sql_start = response.find("Assistant:") + len("Assistant:")
            sql_query = response[sql_start:].strip()
            
            results.append({
                "query": query,
                "generated_sql": sql_query,
                "full_response": response
            })
        
        return {"evaluation_results": results}


def main():
    """Main training function."""
    trainer = ARGOModelTrainer()
    
    # Train the model
    model_path = trainer.train_model()
    
    # Test queries
    test_queries = [
        "Show me salinity profiles in the Arabian Sea",
        "Find temperature data in the Indian Ocean",
        "What are the ARGO float trajectories near the equator?"
    ]
    
    # Evaluate
    results = trainer.evaluate_model(test_queries)
    
    # Save evaluation results
    with open("argo_model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Training completed!")
    print(f"Model saved to: {model_path}")
    print("Evaluation results saved to: argo_model_evaluation.json")


if __name__ == "__main__":
    main()






