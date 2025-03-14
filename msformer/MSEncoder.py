import os
import yaml
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule
from fast_transformers.masking import LengthMask as LM
from tqdm import tqdm
from typing import List, Union, Optional

class MSEncoder:
    """Molecular Substructure Encoder for chemical fragment representation learning
    
    Attributes:
        model (LightningModule): Pretrained transformer model
        tokenizer (MolTranBertTokenizer): Chemical tokenizer
        device (torch.device): Computation device (GPU/CPU)
        max_fragments (int): Maximum number of fragments per molecule
        embed_dim (int): Dimension of fragment embeddings
    
    Methods:
        encode(fragments: List[List[str]]) -> np.ndarray
        save_embeddings(embeddings: np.ndarray, output_path: str)
    """
    
    def __init__(self, 
                 config_path: str, 
                 checkpoint_path: str, 
                 vocab_file: str = 'bert_vocab.txt',
                 device: str = 'cuda',
                 max_fragments: int = 2000,
                 embed_dim: int = 768):
        """Initialize molecular substructure encoder
        
        Args:
            config_path (str): Path to model configuration YAML file
            checkpoint_path (str): Path to pretrained model checkpoint
            vocab_file (str): Path to tokenizer vocabulary file
            device (str): Computation device ('cuda' or 'cpu')
            max_fragments (int): Maximum fragments per molecule for padding
            embed_dim (int): Dimension of fragment embeddings
        """
        self.device = torch.device(device)
        self.max_fragments = max_fragments
        self.embed_dim = embed_dim
        
        # Load model configuration
        with open(config_path, 'r') as f:
            self.config = Namespace(**yaml.safe_load(f))
            
        # Initialize tokenizer and model
        self.tokenizer = MolTranBertTokenizer(vocab_file)
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path: str) -> LightningModule:
        """Load pretrained model from checkpoint"""
        model = LightningModule.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
            vocab=self.tokenizer.vocab
        )
        model.eval()
        return model.to(self.device)
    
    def _batch_generator(self, data: list, batch_size: int = 64):
        """Generate mini-batches from input data"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def _encode_batch(self, batch: List[str]) -> torch.Tensor:
        """Encode a batch of fragments"""
        encodings = self.tokenizer.batch_encode_plus(
            batch, 
            padding=True, 
            add_special_tokens=True
        )
        
        input_ids = torch.tensor(encodings['input_ids']).to(self.device)
        attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.blocks(
                self.model.tok_emb(input_ids),
                length_mask=LM(attention_mask.sum(-1))
            )
            
        # Mask-aware mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
        sum_embeddings = torch.sum(outputs * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu()
    
    def encode(self, 
              fragments: List[List[str]], 
              batch_size: int = 64,
              verbose: bool = True) -> np.ndarray:
        """Encode molecular fragments into embeddings
        
        Args:
            fragments (List[List[str]]): List of molecules, each represented 
                as a list of SMILES fragments
            batch_size (int): Number of fragments per processing batch
            verbose (bool): Show progress bar
            
        Returns:
            np.ndarray: 3D tensor of shape (molecules, max_fragments, embed_dim)
        """
        embeddings = np.zeros((len(fragments), self.max_fragments, self.embed_dim))
        
        iterator = tqdm(fragments, desc="Encoding molecules") if verbose else fragments
        
        for idx, mol_fragments in enumerate(iterator):
            try:
                batch_embeds = []
                for batch in self._batch_generator(mol_fragments, batch_size):
                    batch_embeds.append(self._encode_batch(batch).numpy())
                
                mol_embedding = np.concatenate(batch_embeds, axis=0)
                num_frags = min(mol_embedding.shape[0], self.max_fragments)
                embeddings[idx, :num_frags, :] = mol_embedding[:num_frags]
                
            except Exception as e:
                print(f"Error processing molecule {idx}: {str(e)}")
                continue
                
        return embeddings
    
    def save_embeddings(self, 
                       embeddings: np.ndarray, 
                       output_path: str, 
                       compression: Optional[str] = None):
        """Save embeddings to disk
        
        Args:
            embeddings (np.ndarray): Fragment embeddings array
            output_path (str): Output file path
            compression (str): Compression method for np.savez (e.g., 'npz')
        """
        if compression:
            np.savez_compressed(output_path, embeddings=embeddings)
        else:
            np.save(output_path, embeddings)
