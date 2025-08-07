# graphysics

**Graph Neural Networks for Physics Equation Discovery using Enhanced Semantic Representations**

## Quick Start

**Option 1: Local Python (Recommended for Reproducibility)**
```bash
git clone https://github.com/kingelanci/graphysics.git
cd graphysics
pip install -r requirements.txt
jupyter notebook SemanticGNN-Physics.ipynb
```

**Option 2: Google Colab (Convenience only)**
Click "Open in Colab" badge above

⚠️ **Note**: For full deterministic reproducibility, use local Python environment. Colab may have slight variations due to hardware/software differences.

## Results

| Model | File | Epochs | AUC | AP |
|-------|------|--------|-----|-----|
| SuperGNN-Fast       | `grm_trainer_1000.py`  | 1000 | **0.9548 ± 0.0071** | **0.9475 ± 0.0074** |
| SuperGNN-Deep       | `grm_trainer_3000.py`  | 3000 | **0.9706 ± 0.0036** | **0.9658 ± 0.0042** |
| SuperGNN-comparison | `models_comparison.py` | 5000 | **0.9746 ± 0.0026** | **0.9697 ± 0.0032** |

**5 Independent Runs**: Seeds [42, 123, 456, 789, 999] - Full reproducibility  
**Performance**: +1.58% AUC improvement from 1000→3000 epochs, significant better than other models 5000 epochs (models_comparison.py)

## Pipeline Architecture
**SuperPhysicsGNN**: 52,065 parameters, Multi-Head GAT with Edge Weights
- **Attention Heads**: 4 → 2 → 1  
- **Hidden Dims**: 64 → 32 → 16
- **Edge Attributes**: 3D (normalized_weight, fundamental_weight, quality_score)

1. **AST Processing**: `ast_parser.py` - 400 equations parsed (100% success rate)
2. **Graph Building**: `graph_builder.py` - 639 nodes, 11,806 edges, 4,557 equation bridges  
3. **GNN Training**: `grm_trainer_1000.py` or `grm_trainer_3000.py` - SuperPhysicsGNN
4. **Analysis**: Cross-domain physics discovery with attention mechanism

## Key Results from Experiments
- **Graph Construction**: 10,422 bridges found → 4,557 significant kept
- **Training Time**: ~15 minutes (1000 epochs) or ~45 minutes (3000 epochs) on CPU
- **Architecture**: Multi-Head GAT with Physics-Aware Decoder
- **Reproducibility**: 5 independent runs with different seeds
- **Top Equation Hub**: Compton Scattering (92 connections, avg weight 0.704)

- Statistical Validation

FDR Analysis (fdr_analysis.py) - Bootstrap confidence intervals, negative controls, sensitivity analysis
Model Comparison (model_comparison.py) - Statistical testing against baselines (GCN, GAT, GraphSAGE)

Network Visualizations

Ego Networks (ego_networks.py) - Physics concept-centered visualizations

## Requirements

```
torch>=2.0.0
torch-geometric>=2.3.0  
networkx>=3.1
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
sympy>=1.12
tqdm>=4.65.0
scipy>=1.10.0
plotly>=5.0.0
```

## Files Structure
```
graphysics/
├── SemanticGNN-Physics.ipynb          # Complete pipeline notebook
├── ast_parser.py                      # AST processing
├── graph_builder.py                   # Graph construction  
├── grm_trainer_1000.py               # Fast training (15min)
├── grm_trainer_3000.py               # Full training (45min)
├── final_physics_database.json       # 400 equations dataset
├── requirements.txt                   # Dependencies
└── analysis/
    ├── fdr_analysis.py               # Statistical FDR validation
    ├── ego_networks.py               # Graph visualizations & ego networks
    └── model_comparison.py           # Baseline comparisons
```

## Author
Massimiliano Romiti  
Indipendent Researcher
 
📧 massimiliano.romiti@acm.org  
🔗 [ORCID](https://orcid.org/0009-0004-7264-9703)

## Citation
```bibtex
@article{graphysics2025,
  title={Graph Neural Networks for Physics Equation Discovery using Enhanced Semantic Representations},
  author={Massimiliano Romiti},
  year={2025},
  doi={https://doi.org/10.5281/zenodo.16530467}
}
```
