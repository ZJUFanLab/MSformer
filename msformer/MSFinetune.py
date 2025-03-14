"""
MSformer: Meta Structure-aware Transformer for Molecular Property Prediction

Implementation of a transformer-based architecture for learning molecular 
representations from mass spectrometry-inspired fragmentation patterns.
"""

# Standard library imports
import os
import time
import random
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Dict, Any, Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from fast_transformers.masking import LengthMask
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

class ResidualMLP(nn.Module):
    """Multi-layer Perceptron with optional residual connections
    
    Args:
        config: Namespace containing model parameters:
            - n_embd: Embedding dimension
            - num_classes: Number of output classes
            - dropout: Dropout probability
    
    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        final: Output projection layer
    """
    
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.desc_skip_connection = False
        
        # Layer definitions
        self.fc1 = nn.Linear(config.n_embd, config.n_embd)
        self.dropout1 = nn.Dropout(config.dropout)
        self.activation1 = nn.GELU()
        
        self.fc2 = nn.Linear(config.n_embd, config.n_embd)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation2 = nn.GELU()
        
        self.final = nn.Linear(config.n_embd, config.num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights using normal distribution"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional residual connections"""
        identity = x
        
        # First MLP block
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation1(x)
        
        # Residual connection
        if self.desc_skip_connection:
            x += identity
        
        # Second MLP block
        z = self.fc2(x)
        z = self.dropout2(z)
        z = self.activation2(z)
        
        # Final projection
        output = self.final(z + x) if self.desc_skip_connection else self.final(z)
        
        return output, z

class MolecularTransformer(pl.LightningModule):
    """Transformer-based model for molecular property prediction
    
    Implements:
    - Multi-head self-attention with generalized random features
    - Adaptive optimization with LAMB optimizer
    - Comprehensive metric tracking
    
    Args:
        config: Namespace containing model hyperparameters
    """
    
    def __init__(self, config: Namespace):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Transformer encoder
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd // config.n_head,
            value_dimensions=config.n_embd // config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type=config.attention_type,
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu'
        )
        self.encoder = builder.get()
        self.dropout = nn.Dropout(config.d_dropout)
        
        # Prediction head
        self.predictor = ResidualMLP(config)
        
        # Metrics storage
        self.training_metrics = []
        self.validation_metrics = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure LAMB optimizer with weight decay filtering"""
        decay_params, no_decay_params = self._get_parameter_groups()
        
        optimizer = optimizers.FusedLAMB(
            [
                {"params": decay_params, "weight_decay": 0.0},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr_start * self.hparams.lr_multiplier,
            betas=(0.9, 0.999) if self.hparams.measure_name == 'r2' else (0.9, 0.99)
        )
        return optimizer

    def _get_parameter_groups(self) -> Tuple[List, List]:
        """Separate parameters into decay/no-decay groups"""
        decay = set()
        no_decay = set()
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    if isinstance(m, nn.Linear):
                        decay.add(fpn)
                    elif isinstance(m, (nn.LayerNorm, nn.Embedding)):
                        no_decay.add(fpn)
        
        # Validate parameter partitioning
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert not inter_params, f"Conflicting params: {inter_params}"
        assert not (param_dict.keys() - union_params), f"Unassigned params: {param_dict.keys() - union_params}"
        
        return [param_dict[pn] for pn in sorted(decay)], [param_dict[pn] for pn in sorted(no_decay)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base forward pass through transformer encoder"""
        return self.encoder(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step with loss calculation and metric logging"""
        inputs, targets = batch
        embeddings = self.forward(inputs)
        embeddings = torch.mean(embeddings, dim=1)
        
        logits, _ = self.predictor(embeddings)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return {'loss': loss, 'embeddings': embeddings.detach()}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step with comprehensive metric calculation"""
        inputs, targets = batch
        embeddings = self.forward(inputs)
        embeddings = torch.mean(embeddings, dim=1)
        
        logits, _ = self.predictor(embeddings)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        return {
            'val_loss': loss,
            'preds': preds,
            'targets': targets,
            'probs': probs[:, 1]  # For binary classification
        }

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        """Aggregate validation metrics and log results"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        preds = torch.cat([x['preds'] for x in outputs]).cpu()
        targets = torch.cat([x['targets'] for x in outputs]).cpu()
        probs = torch.cat([x['probs'] for x in outputs]).cpu()
        
        metrics = {
            'val_loss': avg_loss,
            'val_acc': accuracy_score(targets, preds),
            'val_auroc': roc_auc_score(targets, probs),
            'val_recall': recall_score(targets, preds, average='micro')
        }
        
        self.log_dict(metrics, prog_bar=True)
        self._save_validation_results(metrics)

    def _save_validation_results(self, metrics: Dict[str, float]):
        """Save validation results to CSV for reproducibility"""
        os.makedirs(self.hparams.results_dir, exist_ok=True)
        results_path = os.path.join(self.hparams.results_dir, "validation_metrics.csv")
        
        pd.DataFrame([metrics]).to_csv(
            results_path,
            mode='a',
            header=not os.path.exists(results_path)
        )

class MolecularDataset(Dataset):
    """Dataset for molecular fragment representations
    
    Args:
        fragment_data: Tensor of shape (n_samples, max_fragments, n_features)
        labels: Tensor of shape (n_samples,)
        min_valid_fragments: Minimum number of valid fragments per sample
    
    Attributes:
        data: Filtered fragment data tensor
        labels: Corresponding labels
        attention_mask: Binary mask indicating valid fragments
    """
    
    def __init__(self, 
                 fragment_data: torch.Tensor, 
                 labels: torch.Tensor,
                 min_valid_fragments: int = 1):
        super().__init__()
        
        # Data validation
        assert fragment_data.size(0) == labels.size(0), \
            "Data and labels must have same first dimension"
            
        # Filter invalid samples
        valid_samples = self._filter_samples(fragment_data, labels, min_valid_fragments)
        self.data = fragment_data[valid_samples]
        self.labels = labels[valid_samples]
        
        # Generate attention masks
        self.attention_mask = self._create_attention_masks(self.data)
        
    def _filter_samples(self, 
                       data: torch.Tensor, 
                       labels: torch.Tensor,
                       min_frags: int) -> torch.Tensor:
        """Filter samples with insufficient valid fragments or NaN labels"""
        valid_fragments = (data.abs().sum(dim=-1) > 0).sum(dim=1) >= min_frags
        valid_labels = ~torch.isnan(labels)
        return valid_fragments & valid_labels
    
    def _create_attention_masks(self, data: torch.Tensor) -> torch.Tensor:
        """Generate binary masks indicating non-zero fragments"""
        return (data.abs().sum(dim=-1) > 0).float()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.data[idx], 
            self.labels[idx],
            self.attention_mask[idx]
        )

class MolecularDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for molecular data handling
    
    Args:
        data_root: Path to data directory
        batch_size: Number of samples per batch
        num_workers: CPU threads for data loading
        max_fragments: Maximum fragments per molecule
        measure_name: Target property name in CSV files
    """
    
    def __init__(self, 
                 data_root: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 max_fragments: int = 1000,
                 measure_name: str = "activity"):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Load and preprocess datasets"""
        # Load raw data
        train_data, train_labels = self._load_dataset("train")
        val_data, val_labels = self._load_dataset("valid")
        test_data, test_labels = self._load_dataset("test")
        
        # Create datasets
        self.train_dataset = MolecularDataset(train_data, train_labels)
        self.val_dataset = MolecularDataset(val_data, val_labels)
        self.test_dataset = MolecularDataset(test_data, test_labels)
        
    def _load_dataset(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load numpy arrays and CSV labels for a data split"""
        data_path = os.path.join(self.hparams.data_root, f"frag{split}_X.npy")
        label_path = os.path.join(self.hparams.data_root, f"frag{split}.csv")
        
        # Load and process data
        data = torch.from_numpy(np.load(data_path)[:, :self.hparams.max_fragments, :])
        labels = pd.read_csv(label_path)[self.hparams.measure_name].values
        labels = torch.from_numpy(labels).float()
        
        return data, labels
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )


def run_experiment(config: Namespace) -> Dict[str, Any]:
    """Execute complete training workflow
    
    Args:
        config: Experiment configuration containing:
            - data_root: Path to data directory
            - checkpoints_folder: Model saving directory
            - expname: Experiment identifier
            - max_epochs: Training epochs
            - seed: Random seed
            
    Returns:
        Dictionary containing training metrics and model paths
    """
    
    # Set reproducibility
    pl.seed_everything(config.seed)
    
    try:
        # Initialize data and model
        datamodule = MolecularDataModule(config.data_root)
        model = MolecularTransformer(config)
        
        # Configure logging and checkpoints
        logger, callbacks = _setup_experiment_tracking(config)
        
        # Train model
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator="auto",
            devices="auto",
            enable_progress_bar=True,
            deterministic=True
        )
        
        # Execute training
        start_time = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        training_time = time.perf_counter() - start_time
        
        # Final evaluation
        test_results = trainer.test(datamodule=datamodule)[0]
        
        return {
            "training_time": training_time,
            "test_metrics": test_results,
            "checkpoint_path": trainer.checkpoint_callback.best_model_path,
            "log_dir": logger.log_dir
        }
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise

def _setup_experiment_tracking(config: Namespace) -> Tuple[pl.Logger, list]:
    """Configure experiment tracking components"""
    
    # Checkpoint directory
    checkpoint_dir = os.path.join(config.checkpoints_folder, config.expname, "models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True
    )
    
    # Logging
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(config.checkpoints_folder, config.expname),
        name="logs"
    )
    
    return logger, [checkpoint_callback, early_stopping]

if __name__ == "__main__":
    # config.py
    from config_module import ConfigParser
    # initialize configuration 
    config = ConfigParser.parse_args()
    # Run experiment
    results = run_experiment(config)
    # Save final results
    with open(os.path.join(results["log_dir"], "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
