#!/usr/bin/env python3
"""
SUPER GNN PRO - ULTIMATE PHYSICS KNOWLEDGE GRAPH NEURAL NETWORK
==============================================================
Il modello definitivo che sfrutta i ponti pesati tra equazioni
per scoprire connessioni profonde nella fisica.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds - MODIFICA PER SEED PARAMETRICO
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class EdgeWeightedGATConv(nn.Module):
    """
    GAT layer modificato per usare i pesi degli archi come modulatori dell'attenzione.
    Questa è la chiave del Super GNN!
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 8,
                 dropout: float = 0.2, edge_dim: int = 3):
        super().__init__()
        self.gat = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            edge_dim=edge_dim  # CRUCIALE: usa attributi degli archi!
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # La magia: GATConv userà edge_attr per modulare l'attenzione
        out, attention_weights = self.gat(x, edge_index, edge_attr, return_attention_weights=True)
        return out, attention_weights

class SuperPhysicsGNN(nn.Module):
    """
    Il Super GNN definitivo con:
    - Multi-head attention che sfrutta i pesi degli archi
    - Architettura profonda con residual connections
    - Layer specifici per catturare pattern fisici
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128],
                 heads: List[int] = [8, 4, 2], dropout: float = 0.2,
                 edge_dim: int = 3, use_edge_weights: bool = True):
        super().__init__()

        self.use_edge_weights = use_edge_weights
        self.attention_weights_history = []

        # Input projection con layer norm
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Stack di GAT layers con edge weights
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_dim = hidden_dims[0]
        for i, (hidden_dim, n_heads) in enumerate(zip(hidden_dims, heads)):
            # Ogni layer GAT usa i pesi degli archi
            self.gat_layers.append(
                EdgeWeightedGATConv(
                    prev_dim if i == 0 else prev_dim * heads[i-1],
                    hidden_dim,
                    heads=n_heads,
                    dropout=dropout,
                    edge_dim=edge_dim if use_edge_weights else 0
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim * n_heads))
            self.dropouts.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output projection
        final_dim = hidden_dims[-1] * heads[-1]
        self.output_projection = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, hidden_dims[-1])
        )

        # Skip connection dalla input
        self.skip_projection = nn.Linear(hidden_dims[0], hidden_dims[-1])

        # Physics-aware pooling per catturare pattern globali
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Input processing
        x = self.input_norm(x)
        x = self.input_projection(x)
        initial_x = x

        # Pulisci history attenzione
        self.attention_weights_history = []

        # Stack di GAT layers con edge weights
        for i, (gat_layer, layer_norm, dropout) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.dropouts)
        ):
            # GAT con edge weights
            if self.use_edge_weights and edge_attr is not None:
                x, attn_weights = gat_layer(x, edge_index, edge_attr)
                # Salva pesi attenzione per analisi
                self.attention_weights_history.append(attn_weights)
            else:
                x, attn_weights = gat_layer(x, edge_index, None)

            x = layer_norm(x)
            x = F.elu(x)
            x = dropout(x)

        # Output projection
        x = self.output_projection(x)

        # Skip connection
        skip = self.skip_projection(initial_x)
        x = x + skip

        # L2 normalization per stabilità
        x = F.normalize(x, p=2, dim=1)

        return x

    def get_attention_weights(self) -> List[torch.Tensor]:
        """Ritorna i pesi di attenzione per analisi."""
        return self.attention_weights_history

class PhysicsAwareDecoder(nn.Module):
    """
    Decoder avanzato che considera la natura fisica delle connessioni.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Rete per combinare embeddings
        self.combine_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Matrice bilineare per catturare interazioni
        self.W = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index

        # Combinazione non lineare
        z_combined = torch.cat([z[row], z[col]], dim=1)
        score_nonlinear = self.combine_net(z_combined).squeeze()

        # Termine bilineare
        score_bilinear = (z[row] @ self.W * z[col]).sum(dim=1)

        # Combina i two scores
        return score_nonlinear + 0.5 * score_bilinear

class SuperGNNModel(nn.Module):
    """Modello completo con encoder e decoder avanzati."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128],
                 heads: List[int] = [8, 4, 2], edge_dim: int = 3):
        super().__init__()

        self.encoder = SuperPhysicsGNN(
            input_dim, hidden_dims, heads,
            edge_dim=edge_dim, use_edge_weights=True
        )
        self.decoder = PhysicsAwareDecoder(hidden_dims[-1])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encoder(x, edge_index, edge_attr)
        return z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_index)

class SuperGNNTrainer:
    """
    Trainer avanzato con tutte le ottimizzazioni moderne.
    """

    def __init__(self, model: SuperGNNModel, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.history = defaultdict(list)
        self.best_model_state = None
        self.attention_analysis = []

        logger.info(f"Model on device: {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, data: Data, train_pos_edge_index: torch.Tensor,
                   optimizer: torch.optim.Optimizer, batch_size: int = 4096) -> float:
        """Training con edge attributes."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Shuffle
        perm = torch.randperm(train_pos_edge_index.size(1))
        train_pos_edge_index = train_pos_edge_index[:, perm]

        for start_idx in range(0, train_pos_edge_index.size(1), batch_size):
            optimizer.zero_grad()

            end_idx = min(start_idx + batch_size, train_pos_edge_index.size(1))
            pos_edge_batch = train_pos_edge_index[:, start_idx:end_idx]

            # Negative sampling bilanciato
            neg_edge_batch = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edge_batch.size(1) * 2,
                method='sparse'
            )

            # Forward con edge attributes
            z = self.model(data.x, data.edge_index, data.edge_attr)

            # Predictions
            pos_pred = self.model.decode(z, pos_edge_batch)
            neg_pred = self.model.decode(z, neg_edge_batch)

            # Loss con label smoothing
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_pred, torch.ones_like(pos_pred) * 0.95  # Label smoothing
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_pred, torch.zeros_like(neg_pred) + 0.05  # Label smoothing
            )

            loss = pos_loss + neg_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, data: Data, pos_edge_index: torch.Tensor,
                neg_edge_index: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Valutazione con metriche avanzate."""
        self.model.eval()

        # Forward with edge attributes
        z = self.model(data.x, data.edge_index, data.edge_attr)

        # Positive predictions
        pos_pred = torch.sigmoid(self.model.decode(z, pos_edge_index))

        # Negative sampling if not provided
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edge_index.size(1),
                method='sparse'
            )

        neg_pred = torch.sigmoid(self.model.decode(z, neg_edge_index))

        # Metrics
        y_true = torch.cat([
            torch.ones(pos_pred.size(0)),
            torch.zeros(neg_pred.size(0))
        ]).cpu().numpy()

        y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()

        metrics = {
            'auc': roc_auc_score(y_true, y_pred),
            'ap': average_precision_score(y_true, y_pred),
            'acc': ((y_pred > 0.5) == y_true).mean(),
            'pos_acc': (pos_pred.cpu().numpy() > 0.5).mean(),
            'neg_acc': (neg_pred.cpu().numpy() <= 0.5).mean()
        }

        # Precision at different recall levels
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            idx = np.searchsorted(recall[::-1], r)
            if idx < len(precision):
                metrics[f'p@r{r}'] = precision[::-1][idx]

        return metrics


    def train(self, data: Data, train_edges: torch.Tensor, val_edges: torch.Tensor,
              num_epochs: int = 1000, lr: float = 0.001, patience: int = 100,
              batch_size: int = 4096):
        """Training completo con tutti i bells and whistles."""

        data = data.to(self.device)
        train_edges = train_edges.to(self.device)
        val_edges = val_edges.to(self.device)

        # Ottimizzatore avanzato
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=5e-4,
            betas=(0.9, 0.999)
        )

        # Scheduler sofisticato
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

        best_val_auc = 0
        patience_counter = 0

        logger.info(f"Starting Super GNN training for {num_epochs} epochs...")
        logger.info(f"Using edge attributes: {data.edge_attr is not None}")

        pbar = tqdm(range(num_epochs), desc="Training Super GNN")

        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(data, train_edges, optimizer, batch_size)
            self.history['train_loss'].append(train_loss)

            # Validation ogni 5 epoch
            if epoch % 5 == 0:
                val_metrics = self.evaluate(data, val_edges)

                for metric, value in val_metrics.items():
                    self.history[f'val_{metric}'].append(value)

                # Scheduler step
                scheduler.step()

                # Early stopping
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0

                    # Salva analisi attenzione
                    if epoch > 50:  # Solo dopo warmup
                        self._analyze_attention(data)
                else:
                    patience_counter += 1

                # Update progress
                pbar.set_postfix({
                    'loss': f'{train_loss:.4f}',
                    'auc': f'{val_metrics["auc"]:.4f}',
                    'ap': f'{val_metrics["ap"]:.4f}',
                    'best_auc': f'{best_val_auc:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        logger.info(f"Training completed. Best validation AUC: {best_val_auc:.4f}")

        return self.history

    def _analyze_attention(self, data: Data):
        """Analizza i pesi di attenzione per capire cosa ha imparato il modello."""
        self.model.eval()

        with torch.no_grad():
            _ = self.model(data.x, data.edge_index, data.edge_attr)
            attention_weights = self.model.encoder.get_attention_weights()

            if attention_weights:
                # Analizza l'ultimo layer
                last_attention = attention_weights[-1]

                if last_attention is not None:
                    edge_index, weights = last_attention

                    # Trova archi con alta attenzione
                    high_attention_mask = weights.mean(dim=1) > 0.7
                    high_attention_edges = edge_index[:, high_attention_mask]

                    self.attention_analysis.append({
                        'num_high_attention': high_attention_mask.sum().item(),
                        'mean_attention': weights.mean().item(),
                        'std_attention': weights.std().item()
                    })

    @torch.no_grad()
    def predict_new_links(self, data: Data, train_edge_index: torch.Tensor,
                         k: int = 200, focus_on_equations: bool = True,
                         min_quality: float = 0.5) -> List[Tuple[str, str, float, Dict]]:
        """
        Predizione avanzata con focus su equazioni e analisi dell'attenzione.
        """
        self.model.eval()
        data = data.to(self.device)
        train_edge_index = train_edge_index.to(self.device)

        logger.info(f"Generating predictions for all possible new links (focus on equations)...")

        # Get embeddings with edge attributes
        z = self.model(data.x, data.edge_index, data.edge_attr)

        # Identify equation nodes if requested
        if focus_on_equations and hasattr(data, 'y'):
            equation_mask = data.y == 0  # Assuming 0 = equation
            equation_indices = torch.where(equation_mask)[0]
            logger.info(f"Focusing on {len(equation_indices)} equation nodes")
        else:
            equation_indices = torch.arange(data.num_nodes)

        # Set of existing edges
        existing_edges = set()
        for i in range(train_edge_index.size(1)):
            u, v = train_edge_index[0, i].item(), train_edge_index[1, i].item()
            existing_edges.add((min(u, v), max(u, v)))

        # Generate predictions for all potential new links
        all_predictions = []

        # Batch processing for efficiency
        batch_size = 100
        total_pairs = len(equation_indices) * (len(equation_indices) - 1) // 2
        processed_pairs = 0

        with tqdm(total=total_pairs, desc="Generating predictions") as pbar:
            batch_pairs = []

            for i, node_i in enumerate(equation_indices):
                for j, node_j in enumerate(equation_indices[i+1:], i+1):
                    u, v = node_i.item(), node_j.item()

                    if (min(u, v), max(u, v)) not in existing_edges:
                        batch_pairs.append([u, v])

                    # Process batch
                    if len(batch_pairs) >= batch_size or (i == len(equation_indices)-1 and j == len(equation_indices[i+1:])-1):
                         if batch_pairs:
                            edge_batch = torch.tensor(batch_pairs, dtype=torch.long).t().to(self.device)
                            scores = torch.sigmoid(self.model.decode(z, edge_batch)).cpu()

                            # Add analysis extra
                            for idx, (u, v) in enumerate(batch_pairs):
                                score = scores[idx].item()

                                # Calculate embedding similarity
                                embedding_sim = F.cosine_similarity(
                                    z[u].unsqueeze(0),
                                    z[v].unsqueeze(0)
                                ).item()

                                prediction = {
                                    'nodes': (u, v),
                                    'score': score,
                                    'embedding_similarity': embedding_sim,
                                    'node_types': (data.y[u].item(), data.y[v].item()) if hasattr(data, 'y') else (None, None)
                                }
                                all_predictions.append(prediction)

                            processed_pairs += len(batch_pairs)
                            pbar.update(len(batch_pairs))
                            batch_pairs = []

        logger.info(f"Generated {len(all_predictions)} potential new links.")

        # Order by score and take top k
        all_predictions.sort(key=lambda x: x['score'], reverse=True)
        top_k_predictions = all_predictions[:k]

        # Convert to output format with names
        final_predictions = []
        idx_to_node = getattr(data, 'idx_to_node', {})

        for pred in top_k_predictions:
            u, v = pred['nodes']

            name_u = idx_to_node.get(u, f"node_{u}")
            name_v = idx_to_node.get(v, f"node_{v}")

            final_predictions.append((
                name_u,
                name_v,
                pred['score'],
                {
                    'embedding_similarity': pred['embedding_similarity'],
                    'node_types': pred['node_types']
                }
            ))

        return final_predictions


    def analyze_predictions(self, predictions: List[Tuple[str, str, float, Dict]],
                           metadata: Dict) -> Dict[str, Any]:
        """
        Analisi super dettagliata delle predizioni.
        """
        analysis = {
            'total_predictions': len(predictions),
            'score_distribution': {},
            'embedding_similarity_distribution': {},
            'top_equation_connections': [],
            'cross_branch_insights': [],
            'attention_insights': self.attention_analysis,
            'discovery_candidates': []
        }

        # Distribuzioni
        scores = [p[2] for p in predictions]
        similarities = [p[3]['embedding_similarity'] for p in predictions]

        analysis['score_distribution'] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'percentiles': {
                '50': np.percentile(scores, 50),
                '75': np.percentile(scores, 75),
                '90': np.percentile(scores, 90),
                '95': np.percentile(scores, 95),
                '99': np.percentile(scores, 99)
            }
        }

        analysis['embedding_similarity_distribution'] = {
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'correlation_with_score': np.corrcoef(scores, similarities)[0, 1]
        }

        # Top connections tra equazioni
        node_features = metadata.get('node_features', {})

        equation_connections = []
        for n1, n2, score, extra in predictions[:100]:
            if (node_features.get(n1, {}).get('type') == 'equation' and
                node_features.get(n2, {}).get('type') == 'equation'):

                equation_connections.append({
                    'equation1': node_features[n1].get('name', n1),
                    'equation2': node_features[n2].get('name', n2),
                    'branch1': node_features[n1].get('branch', 'Unknown'),
                    'branch2': node_features[n2].get('branch', 'Unknown'),
                    'score': score,
                    'embedding_similarity': extra['embedding_similarity'],
                    'is_cross_branch': node_features[n1].get('branch') != node_features[n2].get('branch')
                })

        analysis['top_equation_connections'] = equation_connections[:100]

        # Insights cross-branch
        cross_branch = [ec for ec in equation_connections if ec['is_cross_branch']]
        if cross_branch:
            branch_pairs = defaultdict(list)
            for conn in cross_branch:
                key = tuple(sorted([conn['branch1'], conn['branch2']]))
                branch_pairs[key].append(conn['score'])

            for (b1, b2), scores in branch_pairs.items():
                analysis['cross_branch_insights'].append({
                    'branches': f"{b1} <-> {b2}",
                    'num_connections': len(scores),
                    'avg_score': np.mean(scores),
                    'max_score': max(scores)
                })

        # Discovery candidates (alta confidenza + alta similarità)
        for n1, n2, score, extra in predictions:
            if score > 0.92 and extra['embedding_similarity'] > 0.5:
                analysis['discovery_candidates'].append({
                    'connection': f"{n1} <-> {n2}",
                    'score': score,
                    'embedding_similarity': extra['embedding_similarity'],
                    'confidence_level': 'VERY HIGH'
                })

        return analysis

    def save_results(self, predictions: List, analysis: Dict, test_metrics: Dict,
                    output_dir: Path = Path('super_gnn_results'), seed: int = 42):  # AGGIUNTO SEED
        """Salva risultati completi con visualizzazioni avanzate."""
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Salva predizioni dettagliate - AGGIUNTO SEED NEL NOME
        with open(output_dir / f'super_predictions_{timestamp}_seed{seed}.json', 'w') as f:
            json.dump({
                'predictions': [
                    {
                        'node1': p[0],
                        'node2': p[1],
                        'score': float(p[2]),
                        'embedding_similarity': float(p[3]['embedding_similarity'])
                    }
                    for p in predictions
                ],
                'analysis': analysis,
                'test_metrics': test_metrics,
                'model_info': {
                    'architecture': 'SuperPhysicsGNN',
                    'uses_edge_weights': True,
                    'attention_layers': 3,
                    'total_parameters': sum(p.numel() for p in self.model.parameters())
                }
            }, f, indent=2)

        # Campionamento stratificato per FDR
        all_scores = [p[2] for p in predictions]
        stratified_predictions = []
        score_bins = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        samples_per_bin = 1000

        for low, high in score_bins:
            bin_preds = [p for p in predictions if low <= p[2] < high]
            if len(bin_preds) > samples_per_bin:
                indices = np.random.choice(len(bin_preds), samples_per_bin, replace=False)
                stratified_predictions.extend([bin_preds[i] for i in indices])
            else:
                stratified_predictions.extend(bin_preds)

        # Salva campione FDR - AGGIUNTO SEED NEL NOME
        with open(output_dir / f'fdr_sample_{timestamp}_seed{seed}.json', 'w') as f:
            json.dump({
                'predictions': [
                    {
                        'node1': p[0],
                        'node2': p[1],
                        'score': float(p[2]),
                        'embedding_similarity': float(p[3]['embedding_similarity'])
                    }
                    for p in stratified_predictions
                ]
            }, f, indent=2)


        # Report dettagliato - AGGIUNTO SEED NEL NOME
        with open(output_dir / f'super_report_{timestamp}_seed{seed}.txt', 'w', encoding='utf-8') as f:
            f.write("SUPER GNN - PHYSICS DISCOVERY REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("MODEL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")

            f.write("\n\nKEY DISCOVERIES - EQUATION CONNECTIONS\n")
            f.write("-"*40 + "\n")
            for i, conn in enumerate(analysis['top_equation_connections'][:100], 1):
                f.write(f"\n{i}. CODE: {n1} <---> {n2}\n")
                f.write(f"   NAME: {conn['equation1']} <---> {conn['equation2']}\n")
                f.write(f"   Score: {conn['score']:.4f}\n")
                f.write(f"   Embedding Similarity: {conn['embedding_similarity']:.4f}\n")
                f.write(f"   Branches: {conn['branch1']} <-> {conn['branch2']}\n")
                f.write(f"   Cross-Branch: {'YES' if conn['is_cross_branch'] else 'NO'}\n")

            f.write("\n\nCROSS-BRANCH INSIGHTS\n")
            f.write("-"*40 + "\n")
            for insight in analysis['cross_branch_insights'][:10]:
                f.write(f"\n{insight['branches']}\n")
                f.write(f"   Connections found: {insight['num_connections']}\n")
                f.write(f"   Average confidence: {insight['avg_score']:.4f}\n")
                f.write(f"   Strongest connection: {insight['max_score']:.4f}\n")

            f.write("\n\nHIGH-CONFIDENCE DISCOVERIES\n")
            f.write("-"*40 + "\n")
            for disc in analysis['discovery_candidates'][:50]:
                f.write(f"\n{disc['connection']}\n")
                f.write(f"   Neural Score: {disc['score']:.4f}\n")
                f.write(f"   Embedding Similarity: {disc['embedding_similarity']:.4f}\n")
                f.write(f"   Status: {disc['confidence_level']}\n")

            if analysis['attention_insights']:
                f.write("\n\nATTENTION MECHANISM INSIGHTS\n")
                f.write("-"*40 + "\n")
                last_attention = analysis['attention_insights'][-1]
                f.write(f"High-attention edges: {last_attention['num_high_attention']}\n")
                f.write(f"Mean attention weight: {last_attention['mean_attention']:.4f}\n")
                f.write(f"Attention std dev: {last_attention['std_attention']:.4f}\n")

        # Visualizzazioni avanzate - AGGIUNTO SEED NEL NOME
        self._create_advanced_visualizations(predictions, analysis, output_dir, timestamp, seed)

        logger.info(f"Results saved to {output_dir}/")

        return output_dir / f'super_report_{timestamp}_seed{seed}.txt'

    def _create_advanced_visualizations(self, predictions, analysis, output_dir, timestamp, seed):  # AGGIUNTO SEED
        """Crea visualizzazioni publication-ready."""
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Score vs Embedding Similarity scatter
        fig, ax = plt.subplots(figsize=(10, 8))

        # Ensure scores and similarities are numpy arrays for correlation calculation
        scores = np.array([p[2] for p in predictions])
        similarities = np.array([p[3]['embedding_similarity'] for p in predictions])

        # Use the first 500 points for plotting for performance
        scores_plot = scores[:500]
        similarities_plot = similarities[:500]

        scatter = ax.scatter(similarities_plot, scores_plot, alpha=0.6, c=scores_plot,
                           cmap='viridis', s=50, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Embedding Cosine Similarity', fontsize=14)
        ax.set_ylabel('Prediction Score', fontsize=14)
        ax.set_title('Neural Score vs Embedding Similarity', fontsize=16, fontweight='bold')

        # Aggiungi colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Prediction Score', fontsize=12)

        # Add regression line using the plotted data
        if len(scores_plot) > 1: # Check if there are enough points to fit a line
            z = np.polyfit(similarities_plot, scores_plot, 1)
            p = np.poly1d(z)
            # Generate points for the regression line across the range of plotted similarities
            x_line = np.linspace(min(similarities_plot), max(similarities_plot), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)


        # Annotations using the full dataset for potentially more representative correlation
        if len(scores) > 1:
            correlation = np.corrcoef(scores, similarities)[0, 1]
            ax.text(0.05, 0.95, f'Correlation (full data): {correlation:.3f}',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_dir / f'score_similarity_analysis_{timestamp}_seed{seed}.png',  # AGGIUNTO SEED
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Cross-branch connection matrix
        if analysis['cross_branch_insights']:
            # Prepara dati per heatmap
            branches = set()
            for insight in analysis['cross_branch_insights']:
                b1, b2 = insight['branches'].split(' <-> ')
                branches.add(b1)
                branches.add(b2)

            branches = sorted(list(branches))
            matrix = np.zeros((len(branches), len(branches)))

            for insight in analysis['cross_branch_insights']:
                b1, b2 = insight['branches'].split(' <-> ')
                i, j = branches.index(b1), branches.index(b2)
                matrix[i, j] = insight['avg_score']
                matrix[j, i] = insight['avg_score']

            # Crea heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(matrix, xticklabels=branches, yticklabels=branches,
                       cmap='YlOrRd', annot=True, fmt='.3f',
                       cbar_kws={'label': 'Average Prediction Score'})

            plt.title('Cross-Branch Connection Strength (Super GNN)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'cross_branch_heatmap_{timestamp}_seed{seed}.png',  # AGGIUNTO SEED
                       dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Training history avanzato
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Validation metrics
        epochs = range(0, len(self.history['val_auc']) * 5, 5)
        axes[0, 1].plot(epochs, self.history['val_auc'], 'o-', label='AUC', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_ap'], 's-', label='AP', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision at different recalls
        recalls = [0.1, 0.3, 0.5, 0.7, 0.9]
        precisions = [self.history.get(f'val_p@r{r}', [])[-1] if self.history.get(f'val_p@r{r}') else 0
                     for r in recalls]

        axes[1, 0].bar(recalls, precisions, color='steelblue', alpha=0.7)
        axes[1, 0].set_xlabel('Recall Level')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision at Different Recall Levels')
        axes[1, 0].grid(True, axis='y', alpha=0.3)

        # Score distribution
        scores = [p[2] for p in predictions]
        axes[1, 1].hist(scores, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
        axes[1, 1].axvline(np.percentile(scores, 90), color='red', linestyle='--',
                          label=f'90th percentile: {np.percentile(scores, 90):.3f}')
        axes[1, 1].set_xlabel('Prediction Score')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Super GNN Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Super GNN Training & Prediction Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'super_gnn_analysis_{timestamp}_seed{seed}.png',  # AGGIUNTO SEED
                   dpi=300, bbox_inches='tight')

        plt.close()


def run_single_seed(seed: int = 42):  # NUOVA FUNZIONE
    """Esegue singolo run con seed specifico."""
    set_seeds(seed)  # IMPOSTA SEED
    
    print(f"🎲 RUN - SEED: {seed}")
    print("="*60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")

    # Trova ultimo grafo
    graph_files = list(Path('knowledge_graph_outputs').glob('physics_knowledge_graph_*.pt'))
    if not graph_files:
        print("❌ ERROR: No knowledge graph found!")
        return

    latest_graph = max(graph_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📊 Loading graph: {latest_graph}")

    # Carica metadata
    timestamp = '_'.join(latest_graph.stem.split('_')[-2:])
    metadata_file = Path('knowledge_graph_outputs') / f'graph_metadata_{timestamp}.json'

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Carica grafo
    data = torch.load(latest_graph, weights_only=False)

    # Verifica edge attributes
    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        print("⚠️ WARNING: No edge attributes found! The graph might not have equation bridges.")
        print("   Run graph3ev.py first to create the enhanced graph with bridges!")
        return

    # Fix attributi mancanti
    if not hasattr(data, 'num_nodes'):
        data.num_nodes = data.x.size(0)

    data.idx_to_node = {int(k): v for k, v in metadata['idx_to_node'].items()}

    print(f"\n📈 Graph Statistics:")
    print(f"   - Total nodes: {data.num_nodes}")
    print(f"   - Total edges: {data.edge_index.size(1) // 2}")
    print(f"   - Node features: {data.x.size(1)}")
    print(f"   - Edge attributes: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
    print(f"   - Equations: {metadata['stats']['equation_nodes']}")
    print(f"   - Concepts: {metadata['stats']['concept_nodes']}")
    print(f"   This will exploit the {metadata['stats'].get('bridge_edges', 0) // 2} equation bridges!")

    # Prepara dati
    edges = to_undirected(data.edge_index)

    # Split (80/10/10)
    perm = torch.randperm(edges.size(1))
    train_size = int(0.8 * edges.size(1))
    val_size = int(0.1 * edges.size(1))

    train_edges = edges[:, perm[:train_size]]
    val_edges = edges[:, perm[train_size:train_size + val_size]]
    test_edges = edges[:, perm[train_size + val_size:]]

    print(f"\n🔄 Data Split:")
    print(f"   - Train edges: {train_edges.size(1) // 2}")
    print(f"   - Val edges: {val_edges.size(1) // 2}")
    print(f"   - Test edges: {test_edges.size(1) // 2}")

    # Inizializza Super GNN
    input_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1) if data.edge_attr is not None else 3

    model = SuperGNNModel(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        heads=[4, 2, 1],
        edge_dim=edge_dim
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 SUPER GNN Architecture:")
    print(f"   - Type: Multi-Head GAT with Edge Weights")
    print(f"   - Attention Heads: 4 → 2 → 1")
    print(f"   - Hidden Dimensions: 64 → 32 → 16")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Edge Attribute Dimensions: {edge_dim}")
    print(f"   - Uses Physics-Aware Decoder: Yes")

    # Inizializza trainer
    trainer = SuperGNNTrainer(model, device)

    # Training
    print(f"\n🚀 Starting SUPER GNN training...")
    print(f"   This will exploit the {metadata['stats'].get('bridge_edges', 0) // 2} equation bridges!")

    history = trainer.train(
        data, train_edges, val_edges,
        num_epochs=3000,
        lr=0.001,
        patience=300,
        batch_size=4096
    )

    print(f"\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(data.to(device), test_edges.to(device))
    print(f"\nTest Performance:")
    for metric, value in test_metrics.items():
        if metric == 'auc':
            # Add confidence interval and p-value for AUC if available
            auc_ci_lower = test_metrics.get('auc_ci_lower', 0.0)
            auc_ci_upper = test_metrics.get('auc_ci_upper', 0.0)
            auc_p_value = test_metrics.get('auc_p_value', 1.0)
            print(f"   - {metric}: {value:.4f} (95% CI: {auc_ci_lower:.4f}-{auc_ci_upper:.4f}, p={auc_p_value:.4e})")
        elif metric not in ['auc_p_value', 'auc_ci_lower', 'auc_ci_upper']:
            print(f"   - {metric}: {value:.4f}")

    # Predizioni avanzate
    print(f"\n🔮 Generating advanced predictions...")
    predictions = trainer.predict_new_links(
        data, train_edges,
        k=100000,  # Più predizioni per analisi profonda
        focus_on_equations=True,
        min_quality=0.6
    )

    # Analisi super dettagliata
    print(f"\n📈 Performing deep analysis...")
    analysis = trainer.analyze_predictions(predictions, metadata)

    # Salva tutto - PASSA SEED
    output_path = trainer.save_results(predictions, analysis, test_metrics, seed=seed)

    # Summary finale
    print(f"\n" + "="*60)
    print(f"✅ RUN COMPLETE! (SEED: {seed})")
    print("="*60)

    print(f"\n📊 FINAL METRICS:")
    print(f"   - Test AUC: {test_metrics['auc']:.4f}")
    print(f"   - Test AP: {test_metrics['ap']:.4f}")
    print(f"   - Precision@Recall=0.9: {test_metrics.get('p@r0.9', 0):.4f}")

    print(f"\n🎯 KEY DISCOVERIES:")
    for i, conn in enumerate(analysis['top_equation_connections'][:5], 1):
        print(f"\n{i}. {conn['equation1'][:40]} <---> {conn['equation2'][:40]}")
        print(f"   Neural Score: {conn['score']:.4f}")
        print(f"   Cross-Branch: {'YES' if conn['is_cross_branch'] else 'NO'}")

    print(f"\n📁 Full report saved to: {output_path}")

    return {
        'seed': seed,
        'auc': test_metrics['auc'],
        'ap': test_metrics['ap'],
        'discoveries': len(analysis['discovery_candidates'])
    }


def main():
    """Esecuzione principale del Super GNN."""
    print("="*80)
    print("SUPER GNN PRO - ULTIMATE PHYSICS KNOWLEDGE DISCOVERY")
    print("="*80)

    # MULTIPLE RUNS CON SEED DIVERSI
    seeds = [42, 123, 456, 789, 999]
    results = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n🎲 RUN #{i}/5 - SEED: {seed}")
        print("="*60)
        
        result = run_single_seed(seed)
        if result:
            results.append(result)
    
    # SUMMARY FINALE
    print(f"\n" + "="*80)
    print("🎉 ALL 5 DETERMINISTIC RUNS COMPLETED!")
    print("="*80)
    
    if results:
        aucs = [r['auc'] for r in results]
        aps = [r['ap'] for r in results]
        
        print(f"\n📊 AGGREGATE METRICS ACROSS SEEDS:")
        print(f"   - AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} (seeds: {seeds})")  # AGGIUNTA RIGA
        print(f"   - AP:  {np.mean(aps):.4f} ± {np.std(aps):.4f} (seeds: {seeds})")   # AGGIUNTA RIGA

    print("\n🚀 The Super GNN has revealed the hidden structure of physics!")


if __name__ == "__main__":
    main()