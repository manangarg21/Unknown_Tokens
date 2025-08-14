from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
import os

class MultilingualSarcasmClassifier(nn.Module):
    """
    Multilingual transformer encoder + classification head with language-specific processing.
    Supports both English and Hindi with appropriate model backbones.
    """
    def __init__(self, model_name: str, num_labels: int = 2, proj_dim: int = 256, 
                 language: str = "en", use_language_embedding: bool = True):
        super().__init__()
        self.language = language
        self.use_language_embedding = use_language_embedding
        
        # Load appropriate model for the language
        if language == "hi":
            # Use Indic-BERT for Hindi
            self.model_name = "ai4bharat/indic-bert"
        else:
            # Use specified model for English
            self.model_name = model_name
            
        self.config = AutoConfig.from_pretrained(self.model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(self.model_name, add_pooling_layer=False)
        
        # Language embedding layer
        if use_language_embedding:
            self.language_embedding = nn.Embedding(2, self.config.hidden_size)  # en=0, hi=1
            self.language_projection = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, proj_dim),
        )
        
        # Attention weights for interpretability
        self.attention_weights = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                language_ids=None, **kwargs):
        # Encode with transformer
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            output_hidden_states=False, 
            output_attentions=True,
            return_dict=True
        )
        
        # Get CLS token representation
        cls = outputs.last_hidden_state[:, 0]
        
        # Store attention weights for interpretability
        if outputs.attentions:
            self.attention_weights = outputs.attentions[-1]  # Last layer attention
        
        # Apply language embedding if enabled
        if self.use_language_embedding and language_ids is not None:
            lang_emb = self.language_embedding(language_ids)
            # Concatenate and project
            combined = torch.cat([cls, lang_emb], dim=-1)
            cls = self.language_projection(combined)
        
        # Apply dropout and classification
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        proj = self.proj(cls)
        
        return {
            "logits": logits, 
            "hidden": cls, 
            "proj": proj,
            "attention_weights": self.attention_weights
        }

class SarcasmClassifier(nn.Module):
    """
    Backward compatibility wrapper for the original model.
    """
    def __init__(self, model_name: str, num_labels: int = 2, proj_dim: int = 256):
        super().__init__()
        self.multilingual_model = MultilingualSarcasmClassifier(
            model_name=model_name, 
            num_labels=num_labels, 
            proj_dim=proj_dim
        )

    def forward(self, **kwargs):
        return self.multilingual_model(**kwargs)

class EnsembleMultilingualClassifier(nn.Module):
    """
    Ensemble of language-specific models for better multilingual performance.
    """
    def __init__(self, english_model_name: str, hindi_model_name: str, 
                 num_labels: int = 2, proj_dim: int = 256):
        super().__init__()
        
        self.english_model = MultilingualSarcasmClassifier(
            english_model_name, num_labels, proj_dim, "en"
        )
        self.hindi_model = MultilingualSarcasmClassifier(
            hindi_model_name, num_labels, proj_dim, "hi"
        )
        
        # Fusion layer
        self.fusion = nn.Linear(self.english_model.config.hidden_size * 2, 
                               self.english_model.config.hidden_size)
        self.classifier = nn.Linear(self.english_model.config.hidden_size, num_labels)
        self.proj = nn.Linear(self.english_model.config.hidden_size, proj_dim)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                language_ids=None, **kwargs):
        # Route to appropriate model based on language
        if language_ids is not None and language_ids[0] == 1:  # Hindi
            outputs = self.hindi_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                language_ids=language_ids
            )
        else:  # English
            outputs = self.english_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                language_ids=language_ids
            )
        
        return outputs

def info_nce_loss(z, pair_idx, temperature: float = 0.07):
    """InfoNCE loss for contrastive learning"""
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / temperature
    B = z.size(0)
    pos = pair_idx
    valid = (pos >= 0)
    if not valid.any():
        return torch.tensor(0.0, device=z.device)
    log_probs = torch.log_softmax(sim, dim=1)
    pos_logp = log_probs[torch.arange(B, device=z.device), torch.where(valid, pos, torch.arange(B, device=z.device))]
    loss = -(pos_logp[valid]).mean()
    return loss

def create_multilingual_model(config: Dict, language: str = "en"):
    """
    Factory function to create appropriate model based on language and config
    """
    lang_config = config.get('languages', {}).get(language, config)
    model_name = lang_config.get('model_name', config.get('model_name', 'roberta-base'))
    
    return MultilingualSarcasmClassifier(
        model_name=model_name,
        num_labels=2,
        proj_dim=256,
        language=language,
        use_language_embedding=True
    )
