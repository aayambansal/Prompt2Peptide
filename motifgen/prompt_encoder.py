#!/usr/bin/env python3
"""
Learned Prompt-to-Constraint Encoder for Prompt2Peptide
Transforms free-text prompts into biophysical constraint predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class PromptEncoder(nn.Module):
    """Learned encoder that maps text prompts to biophysical constraints"""
    
    def __init__(self, 
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Frozen text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Freeze text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Learnable constraint prediction head
        text_dim = self.text_encoder.config.hidden_size
        self.constraint_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8)  # 8 constraint outputs
        )
        
        # Constraint output mapping
        self.constraint_names = [
            'charge_min', 'charge_max', 'muh_min', 'muh_max',
            'gravy_min', 'gravy_max', 'length_min', 'length_max'
        ]
        
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings"""
        inputs = self.tokenizer(prompts, 
                              padding=True, 
                              truncation=True, 
                              max_length=128,
                              return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings
    
    def forward(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Predict constraints from text prompts"""
        text_embeddings = self.encode_text(prompts)
        constraint_logits = self.constraint_head(text_embeddings)
        
        # Split into individual constraints
        constraints = {}
        for i, name in enumerate(self.constraint_names):
            constraints[name] = constraint_logits[:, i]
            
        return constraints
    
    def predict_constraints(self, prompts: List[str]) -> List[Dict[str, float]]:
        """Predict constraints and return as dictionaries"""
        self.eval()
        with torch.no_grad():
            constraints = self.forward(prompts)
            
        results = []
        for i in range(len(prompts)):
            result = {}
            for name in self.constraint_names:
                result[name] = constraints[name][i].item()
            results.append(result)
            
        return results

class PromptConstraintDataset:
    """Dataset for training prompt-to-constraint encoder"""
    
    def __init__(self, data_file: str = None):
        self.prompts = []
        self.constraints = []
        
        if data_file:
            self.load_data(data_file)
        else:
            self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic training data with rule-based mapping"""
        # Define prompt templates and their constraints
        prompt_templates = {
            'cationic amphipathic helix': {
                'charge_min': 3.0, 'charge_max': 8.0,
                'muh_min': 0.35, 'muh_max': 1.0,
                'gravy_min': -0.2, 'gravy_max': 0.6,
                'length_min': 12, 'length_max': 18
            },
            'soluble acidic loop': {
                'charge_min': -3.0, 'charge_max': 0.0,
                'muh_min': 0.1, 'muh_max': 0.4,
                'gravy_min': -1.0, 'gravy_max': 0.0,
                'length_min': 10, 'length_max': 14
            },
            'hydrophobic beta sheet': {
                'charge_min': -1.0, 'charge_max': 2.0,
                'muh_min': 0.1, 'muh_max': 0.3,
                'gravy_min': 0.5, 'gravy_max': 1.5,
                'length_min': 10, 'length_max': 14
            },
            'polar flexible linker': {
                'charge_min': -1.0, 'charge_max': 1.0,
                'muh_min': 0.05, 'muh_max': 0.25,
                'gravy_min': -0.8, 'gravy_max': 0.2,
                'length_min': 8, 'length_max': 12
            },
            'basic nuclear localization': {
                'charge_min': 4.0, 'charge_max': 8.0,
                'muh_min': 0.2, 'muh_max': 0.6,
                'gravy_min': -0.5, 'gravy_max': 0.3,
                'length_min': 7, 'length_max': 12
            }
        }
        
        # Generate variations and paraphrases
        variations = {
            'cationic amphipathic helix': [
                'positive amphipathic helix', 'cationic helical peptide',
                'amphipathic cationic helix', 'positively charged helix',
                'cationic amphipathic alpha helix', 'basic amphipathic helix'
            ],
            'soluble acidic loop': [
                'acidic soluble loop', 'negatively charged loop',
                'soluble acidic region', 'acidic flexible loop',
                'soluble negative loop', 'acidic unstructured region'
            ],
            'hydrophobic beta sheet': [
                'hydrophobic sheet', 'beta sheet hydrophobic',
                'hydrophobic beta strand', 'hydrophobic sheet structure',
                'beta sheet hydrophobic region', 'hydrophobic beta structure'
            ],
            'polar flexible linker': [
                'flexible polar linker', 'polar flexible region',
                'flexible linker polar', 'polar unstructured linker',
                'flexible polar peptide', 'polar flexible sequence'
            ],
            'basic nuclear localization': [
                'nuclear localization signal', 'NLS basic peptide',
                'nuclear targeting signal', 'basic NLS sequence',
                'nuclear localization basic', 'NLS positively charged'
            ]
        }
        
        # Create training data
        for base_prompt, constraints in prompt_templates.items():
            # Add base prompt
            self.prompts.append(base_prompt)
            self.constraints.append(constraints.copy())
            
            # Add variations
            for variation in variations.get(base_prompt, []):
                self.prompts.append(variation)
                # Add small noise to constraints for variation
                noisy_constraints = {}
                for key, value in constraints.items():
                    noise = np.random.normal(0, 0.1 * abs(value))
                    noisy_constraints[key] = value + noise
                self.constraints.append(noisy_constraints)
        
        # Add length variations
        length_variations = []
        for i, (prompt, constraints) in enumerate(zip(self.prompts, self.constraints)):
            if 'length' in prompt.lower():
                # Add length-specific variations
                for length in [8, 10, 12, 15, 18, 20]:
                    new_prompt = f"{prompt}, length {length}"
                    new_constraints = constraints.copy()
                    new_constraints['length_min'] = length - 2
                    new_constraints['length_max'] = length + 2
                    length_variations.append((new_prompt, new_constraints))
        
        # Add length variations to dataset
        for prompt, constraints in length_variations:
            self.prompts.append(prompt)
            self.constraints.append(constraints)
    
    def load_data(self, data_file: str):
        """Load data from file"""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.prompts = data['prompts']
        self.constraints = data['constraints']
    
    def save_data(self, data_file: str):
        """Save data to file"""
        data = {
            'prompts': self.prompts,
            'constraints': self.constraints
        }
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx], self.constraints[idx]

class PromptEncoderTrainer:
    """Trainer for prompt-to-constraint encoder"""
    
    def __init__(self, model: PromptEncoder, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for prompts, target_constraints in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_constraints = self.model.forward(prompts)
            
            # Compute loss for each constraint
            loss = 0.0
            for i, constraint_name in enumerate(self.model.constraint_names):
                target = torch.tensor([tc[constraint_name] for tc in target_constraints], 
                                    dtype=torch.float32)
                loss += self.criterion(predicted_constraints[constraint_name], target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        all_predictions = {name: [] for name in self.model.constraint_names}
        all_targets = {name: [] for name in self.model.constraint_names}
        
        with torch.no_grad():
            for prompts, target_constraints in dataloader:
                predicted_constraints = self.model.forward(prompts)
                
                for constraint_name in self.model.constraint_names:
                    predictions = predicted_constraints[constraint_name].cpu().numpy()
                    targets = [tc[constraint_name] for tc in target_constraints]
                    
                    all_predictions[constraint_name].extend(predictions)
                    all_targets[constraint_name].extend(targets)
        
        # Compute metrics
        metrics = {}
        for constraint_name in self.model.constraint_names:
            pred = np.array(all_predictions[constraint_name])
            target = np.array(all_targets[constraint_name])
            
            metrics[f'{constraint_name}_mae'] = mean_absolute_error(target, pred)
            metrics[f'{constraint_name}_r2'] = r2_score(target, pred)
        
        return metrics

def create_calibration_plots(model: PromptEncoder, test_data, save_path: str):
    """Create calibration plots for constraint predictions"""
    model.eval()
    
    # Get predictions and targets
    prompts, target_constraints = zip(*test_data)
    predictions = model.predict_constraints(prompts)
    
    # Create subplots for each constraint
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    constraint_names = model.constraint_names
    
    for i, constraint_name in enumerate(constraint_names):
        ax = axes[i]
        
        pred_values = [p[constraint_name] for p in predictions]
        target_values = [t[constraint_name] for t in target_constraints]
        
        # Scatter plot
        ax.scatter(target_values, pred_values, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(min(pred_values), min(target_values))
        max_val = max(max(pred_values), max(target_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        # Compute R²
        r2 = r2_score(target_values, pred_values)
        ax.set_title(f'{constraint_name}\nR² = {r2:.3f}')
        ax.set_xlabel('Target')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_prompt_encoder(data_file: str = None, 
                        epochs: int = 50,
                        batch_size: int = 32,
                        save_path: str = 'prompt_encoder.pth'):
    """Train the prompt encoder end-to-end"""
    
    # Create dataset
    dataset = PromptConstraintDataset(data_file)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    
    # Initialize model and trainer
    model = PromptEncoder()
    trainer = PromptEncoderTrainer(model)
    
    # Training loop
    train_losses = []
    val_metrics = []
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metric = trainer.evaluate(val_loader)
        
        train_losses.append(train_loss)
        val_metrics.append(val_metric)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            print(f"Val MAE: {np.mean([v for k, v in val_metric.items() if 'mae' in k]):.4f}")
    
    # Save model
    torch.save(model.state_dict(), save_path)
    
    # Create calibration plots
    create_calibration_plots(model, val_data, 'prompt_encoder_calibration.png')
    
    return model, train_losses, val_metrics

if __name__ == "__main__":
    # Train the prompt encoder
    model, losses, metrics = train_prompt_encoder()
    print("Prompt encoder training completed!")
