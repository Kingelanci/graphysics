# graphysics

**A Graph-Based Framework for Exploring Mathematical Patterns in Physics**

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
**Performance**: +1.58% AUC improvement from 1000→3000 epochs, significantly better than other models at 5000 epochs

## Pipeline Architecture

### 1. Graph Construction & Training
**SuperPhysicsGNN**: 52,065 parameters, Multi-Head GAT with Edge Weights
- **Attention Heads**: 4 → 2 → 1  
- **Hidden Dims**: 64 → 32 → 16
- **Edge Attributes**: 3D (normalized_weight, fundamental_weight, quality_score)

1. **AST Processing**: `ast_parser.py` - 400 equations parsed (100% success rate)
2. **Graph Building**: `graph_builder.py` - 639 nodes, 11,806 edges, 4,557 equation bridges  
3. **GNN Training**: `grm_trainer_1000.py` or `grm_trainer_3000.py` - SuperPhysicsGNN
4. **Analysis**: Cross-domain physics discovery with attention mechanism

### 2. Cluster Analysis & Symbolic Simplification (NEW)
- **Cluster Formation**: `cluster.py` - Identifies equation clusters using multiple algorithms
  - Cliques, Communities (Louvain), K-cores, Connected components
  - Integrates GNN predictions, graph bridges, and variable similarity
  - 30 clusters analyzed (3-99 equations per cluster)
  
- **Symbolic Analysis**: `cluster_analysis.py` - Mathematical simplification engine
  - Backbone equation selection via complexity scoring
  - Variable substitution chains (max 10 iterations)
  - SymPy-based algebraic reduction
  - Classification: IDENTITY, RESIDUAL, SIMPLIFIED, FAILED

## Key Results

### Graph & Training Metrics
- **Graph Construction**: 10,422 bridges found → 4,557 significant kept
- **Training Time**: ~15 minutes (1000 epochs) or ~45 minutes (3000 epochs) on CPU
- **Top Equation Hub**: Compton Scattering (92 connections, avg weight 0.704)

### Symbolic Analysis Results
- **30 clusters analyzed**: 80% produced interpretable results
- **Theory Validation**: Klein-Gordon/Dirac hierarchy confirmed
- **Novel Synthesis**: Magnetic Reynolds Number from EM-fluid coupling
- **Error Detection**: Dimensional inconsistencies identified (c = m_e)
- **Research Directions**: Analog gravity connections discovered

### Statistical Validation
- **FDR Analysis** (`fdr_analysis.py`): Bootstrap confidence intervals, negative controls
- **Model Comparison** (`model_comparison.py`): Statistical testing against baselines

### Network Visualizations
- **Ego Networks** (`ego_networks.py`): Physics concept-centered visualizations
- **Cluster Networks**: Dense mathematical relationship graphs

## Supplementary Materials

Available in `/supplementary/`:
- `all_clusters_report.txt` - Complete analysis of all 30 clusters
- `cluster_simplification_discoveries.txt` - Key symbolic findings and interpretations
- `cluster_simplification_discoveries_test.txt` - Test run results
- `equation_super_prediction_sample_seed123.txt` - Sample predictions with scores

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
├── cluster.py                         # Cluster formation algorithms
├── cluster_analysis.py                # Symbolic simplification engine
├── final_physics_database.json       # 400 equations dataset
├── requirements.txt                   # Dependencies
├── analysis/
│   ├── fdr_analysis.py               # Statistical FDR validation
│   ├── ego_networks.py               # Graph visualizations
│   └── model_comparison.py           # Baseline comparisons
└── supplementary/
    ├── all_clusters_report.txt
    ├── cluster_simplification_discoveries.txt
    ├── cluster_simplification_discoveries_test.txt
    └── equation_super_prediction_sample_seed123.txt
```

## Key Insights

This framework operates as a **hypothesis generation engine**, not a discovery validator. It:
1. **Generates** hundreds of cross-domain mathematical connections
2. **Audits** knowledge bases for consistency and errors
3. **Synthesizes** mathematical relationships across physics domains
4. **Transforms** computational failures into research directions

The system intentionally over-generates candidates (like high-throughput screening) to ensure comprehensive exploration of mathematical possibility space.

## Author
Massimiliano Romiti  
Independent Researcher
 
📧 massimiliano.romiti@acm.org  
🔗 [ORCID](https://orcid.org/0009-0004-7264-9703)

## Citation
```bibtex
@article{romiti2025graphysics,
  title={A Graph-Based Framework for Exploring Mathematical Patterns in Physics: A Proof of Concept},
  author={Romiti, Massimiliano},
  year={2025},
  journal={arXiv preprint arXiv:2508.05724v2}
}
```

## License
MIT License - See LICENSE file for details
