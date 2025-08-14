#!/usr/bin/env python3
"""
MEGA CLUSTER DISCOVERY SYSTEM - VERSIONE DEFINITIVA CON LOGICA GRCOLADSDLWEM
Trova cluster di 3+ equazioni, genera report enriched e visualizzazioni
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurazione matplotlib
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Colori per branche
BRANCH_COLORS = {
    'Quantum Mechanics': '#FF6B6B',
    'Relativity': '#4ECDC4',
    'Thermodynamics': '#45B7D1',
    'Statistical Mechanics': '#96CEB4',
    'Electromagnetism': '#FECA57',
    'Optics': '#DDA0DD',
    'Atomic and Nuclear Physics': '#98D8C8',
    'Condensed Matter Physics': '#FFB6C1',
    'Particle Physics': '#87CEEB',
    'Astrophysics': '#F0E68C',
    'Unknown': '#808080'
}

class MegaClusterDiscovery:
    """Sistema completo per trovare cluster di equazioni"""
    
    def __init__(self, database_file: str = "final_physics_database.json"):
        """Inizializza il sistema"""
        print("="*80)
        print("🚀 MEGA CLUSTER DISCOVERY SYSTEM")
        print("="*80)
        
        self.G = nx.Graph()
        self.equation_map = {}
        self.metadata = None
        self.bridges = None
        self.predictions = None
        self.node_features = {}
        self.equation_nodes = []
        self.concept_nodes = []
        self.idx_to_node = {}
        self.node_to_idx = {}
        
        # Carica database
        self._load_database(database_file)
        
        # Carica altri file se disponibili
        self._load_graph_data()
        
        # Costruisci grafo
        self._build_graph()
    
    def _load_database(self, database_file: str):
        """Carica il database delle equazioni - VERSIONE CORRETTA"""
        try:
            with open(database_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # LA STRUTTURA È: {"metadata": {...}, "constants": [...], "laws": [...]}
            if isinstance(data, dict) and 'laws' in data:
                equations = data['laws']
                print(f"✅ Found {len(equations)} equations in 'laws' key")
                
                # Mostra metadata se presente
                if 'metadata' in data:
                    meta = data['metadata']
                    print(f"   Database version: {meta.get('version', 'Unknown')}")
                
                # Crea mappa delle equazioni
                for eq in equations:
                    if isinstance(eq, dict) and 'id' in eq:
                        self.equation_map[eq['id']] = eq
                
                print(f"✅ Loaded {len(self.equation_map)} equations into map")
            else:
                print("❌ Database file doesn't have expected structure")
                
        except Exception as e:
            print(f"❌ Error loading database: {e}")
    
    def _load_graph_data(self):
        """Carica dati da altri file se disponibili - COME GRCOLADSDLWEM"""
        # Cerca file metadata con node_features
        try:
            metadata_files = list(Path('.').glob('graph_metadata_*.json'))
            if metadata_files:
                latest = max(metadata_files, key=lambda x: x.stat().st_mtime)
                with open(latest, 'r') as f:
                    self.metadata = json.load(f)
                    
                # ESTRAI NODE FEATURES COME GRCOLADSDLWEM
                self.node_features = self.metadata.get('node_features', {})
                self.idx_to_node = {int(k): v for k, v in self.metadata.get('idx_to_node', {}).items()}
                self.node_to_idx = self.metadata.get('node_to_idx', {})
                
                print(f"✅ Loaded metadata with {len(self.node_features)} node features")
                
                # Separa equation nodes da concept nodes
                for node_id, features in self.node_features.items():
                    node_type = features.get('node_type', features.get('type', ''))
                    if node_type == 'equation' or features.get('is_equation'):
                        self.equation_nodes.append(node_id)
                    elif node_type == 'concept':
                        self.concept_nodes.append(node_id)
                
                print(f"   Found {len(self.equation_nodes)} equation nodes")
                print(f"   Found {len(self.concept_nodes)} concept nodes")
        except:
            self.metadata = None
            self.node_features = {}
            self.equation_nodes = []
            self.concept_nodes = []
        
        # Cerca file bridges
        try:
            bridge_files = list(Path('.').glob('equation_bridges_*.json'))
            if bridge_files:
                latest = max(bridge_files, key=lambda x: x.stat().st_mtime)
                with open(latest, 'r') as f:
                    self.bridges = json.load(f)
                print(f"✅ Loaded {len(self.bridges.get('significant_bridges', []))} bridges")
        except:
            self.bridges = None
        
        # Cerca predictions
        try:
            pred_files = list(Path('.').glob('super_predictions_*.json'))
            if pred_files:
                latest = max(pred_files, key=lambda x: x.stat().st_mtime)
                with open(latest, 'r') as f:
                    pred_data = json.load(f)
                    if 'predictions' in pred_data:
                        self.predictions = pred_data['predictions']
                    else:
                        self.predictions = pred_data
                print(f"✅ Loaded {len(self.predictions)} predictions")
        except:
            self.predictions = None
    
    def _build_graph(self):
        """Costruisce il grafo NetworkX - LOGICA COMPLETA GRCOLADSDLWEM"""
        print("\n🔨 Building graph from all sources...")
        
        # Aggiungi nodi (equazioni)
        for eq_id, eq_data in self.equation_map.items():
            self.G.add_node(eq_id, **eq_data)
        
        edges_added = {'bridges': 0, 'predictions': 0, 'variables': 0}
        
        # 1. EDGES DAI BRIDGES (equation-equation bridges da graph3ev)
        if self.bridges:
            print("   Adding edges from equation bridges...")
            for bridge in self.bridges.get('significant_bridges', []):
                # Usa tutti i possibili campi per trovare source e target
                source = self._find_eq_id(bridge.get('source_name', bridge.get('source', bridge.get('source_id'))))
                target = self._find_eq_id(bridge.get('target_name', bridge.get('target', bridge.get('target_id'))))
                
                if source and target and source in self.equation_map and target in self.equation_map:
                    weight = bridge.get('weight', bridge.get('normalized_weight', 1.0))
                    quality = bridge.get('quality_score', 0)
                    
                    # Se edge esiste già, aggiorna peso
                    if self.G.has_edge(source, target):
                        old_weight = self.G[source][target].get('weight', 0)
                        self.G[source][target]['weight'] = max(old_weight, weight)
                        self.G[source][target]['bridge_quality'] = quality
                    else:
                        self.G.add_edge(source, target, 
                                      weight=weight,
                                      bridge_quality=quality,
                                      source='bridges')
                        edges_added['bridges'] += 1
        
        # 2. EDGES DALLE PREDICTIONS (GNN predictions)
        if self.predictions:
            print("   Adding edges from GNN predictions...")
            for pred in self.predictions:
                score = pred.get('score', 0)
                if score > 0.5:  # Soglia più bassa per catturare più cluster
                    # Prova a trovare gli ID
                    node1 = pred.get('node1')
                    node2 = pred.get('node2')
                    
                    # Se sono indici, converti
                    if isinstance(node1, int) and node1 in self.idx_to_node:
                        node1 = self.idx_to_node[node1]
                    if isinstance(node2, int) and node2 in self.idx_to_node:
                        node2 = self.idx_to_node[node2]
                    
                    source = self._find_eq_id(node1)
                    target = self._find_eq_id(node2)
                    
                    if source and target and source in self.equation_map and target in self.equation_map:
                        embed_sim = pred.get('embedding_similarity', 0)
                        
                        # Peso combinato
                        weight = 0.7 * score + 0.3 * embed_sim
                        
                        if self.G.has_edge(source, target):
                            old_weight = self.G[source][target].get('weight', 0)
                            self.G[source][target]['weight'] = max(old_weight, weight)
                            self.G[source][target]['neural_score'] = score
                            self.G[source][target]['embedding_sim'] = embed_sim
                        else:
                            self.G.add_edge(source, target,
                                          weight=weight,
                                          neural_score=score,
                                          embedding_sim=embed_sim,
                                          source='predictions')
                            edges_added['predictions'] += 1
        
        # 3. EDGES BASATI SU VARIABILI CONDIVISE (sempre!)
        print("   Building edges from shared variables...")
        self._build_similarity_edges()
        edges_added['variables'] = self.G.number_of_edges() - edges_added['bridges'] - edges_added['predictions']
        
        # Report
        print(f"\n✅ Graph built:")
        print(f"   Nodes: {self.G.number_of_nodes()} equations")
        print(f"   Total edges: {self.G.number_of_edges()}")
        print(f"   - From bridges: {edges_added['bridges']}")
        print(f"   - From predictions: {edges_added['predictions']}")
        print(f"   - From variables: {edges_added['variables']}")
        
        if self.G.number_of_edges() == 0:
            print("⚠️ Warning: No edges found! Check your data files.")
    
    def _find_eq_id(self, name_or_id):
        """Trova ID equazione - VERSIONE ROBUSTA"""
        if not name_or_id:
            return None
        
        # Se è già un ID valido
        if name_or_id in self.equation_map:
            return name_or_id
        
        # Cerca per nome esatto
        for eq_id, eq_data in self.equation_map.items():
            if eq_data.get('name') == name_or_id:
                return eq_id
        
        # Se è un indice numerico dal metadata
        if self.idx_to_node and isinstance(name_or_id, (int, str)):
            try:
                idx = int(name_or_id) if isinstance(name_or_id, str) else name_or_id
                if idx in self.idx_to_node:
                    node = self.idx_to_node[idx]
                    # Verifica che sia un'equazione
                    if node in self.equation_map:
                        return node
            except:
                pass
        
        # Cerca match parziale nel nome (ultima risorsa)
        if isinstance(name_or_id, str):
            for eq_id, eq_data in self.equation_map.items():
                eq_name = eq_data.get('name', '')
                if name_or_id in eq_name or eq_name in name_or_id:
                    return eq_id
        
        return None
    
    def _build_similarity_edges(self):
        """Costruisce edges basati su similarità delle variabili"""
        print("   Building edges from variable similarity...")
        
        edge_count_before = self.G.number_of_edges()
        
        for eq1_id, eq1 in self.equation_map.items():
            vars1 = set(eq1.get('variables', []))
            if not vars1:
                continue
                
            for eq2_id, eq2 in self.equation_map.items():
                if eq1_id >= eq2_id:  # Evita duplicati
                    continue
                    
                vars2 = set(eq2.get('variables', []))
                if not vars2:
                    continue
                
                # Jaccard similarity
                intersection = vars1.intersection(vars2)
                union = vars1.union(vars2)
                
                if len(intersection) >= 2:  # Almeno 2 variabili in comune
                    similarity = len(intersection) / len(union)
                    if similarity > 0.3:
                        # Se l'edge non esiste già
                        if not self.G.has_edge(eq1_id, eq2_id):
                            self.G.add_edge(eq1_id, eq2_id,
                                          weight=similarity,
                                          shared_vars=list(intersection),
                                          source='similarity')
        
        edge_count_after = self.G.number_of_edges()
        print(f"   Added {edge_count_after - edge_count_before} similarity edges")
    
    def find_clusters(self, min_size: int = 3):
        """Trova tutti i cluster"""
        print(f"\n🔍 Finding clusters (min size: {min_size})...")
        
        clusters = {
            'cliques': [],
            'communities': [],
            'k_cores': [],
            'components': []
        }
        
        # 1. Cliques
        try:
            cliques = list(nx.find_cliques(self.G))
            for clique in cliques:
                if len(clique) >= min_size:
                    clusters['cliques'].append({
                        'nodes': clique,
                        'size': len(clique),
                        'type': 'clique'
                    })
            print(f"   Found {len(clusters['cliques'])} cliques")
        except:
            print("   Could not find cliques")
        
        # 2. Communities (Louvain)
        try:
            communities = nx.community.louvain_communities(self.G, weight='weight', seed=42)
            for comm in communities:
                if len(comm) >= min_size:
                    clusters['communities'].append({
                        'nodes': list(comm),
                        'size': len(comm),
                        'type': 'community'
                    })
            print(f"   Found {len(clusters['communities'])} communities")
        except:
            print("   Could not find communities")
        
        # 3. K-cores
        try:
            max_degree = max(dict(self.G.degree()).values()) if self.G.degree() else 0
            for k in range(2, min(20, max_degree + 1)):
                k_core = nx.k_core(self.G, k)
                if k_core.number_of_nodes() >= min_size:
                    for comp in nx.connected_components(k_core):
                        if len(comp) >= min_size:
                            clusters['k_cores'].append({
                                'nodes': list(comp),
                                'size': len(comp),
                                'type': f'{k}-core',
                                'k': k
                            })
            print(f"   Found {len(clusters['k_cores'])} k-cores")
        except:
            print("   Could not find k-cores")
        
        # 4. Connected components
        for comp in nx.connected_components(self.G):
            if len(comp) >= min_size:
                clusters['components'].append({
                    'nodes': list(comp),
                    'size': len(comp),
                    'type': 'component'
                })
        print(f"   Found {len(clusters['components'])} components")
        
        return clusters
    
    def analyze_cluster(self, cluster):
        """Analizza un cluster in dettaglio con metriche GRCOLADSDLWEM"""
        nodes = cluster['nodes']
        subgraph = self.G.subgraph(nodes)
        
        analysis = {
            'size': len(nodes),
            'type': cluster['type'],
            'density': nx.density(subgraph) if len(nodes) > 1 else 0,
            'equations': [],
            'branches': Counter(),
            'shared_concepts': Counter(),
            'connections': [],
            'cluster_metrics': {}
        }
        
        # Calcola metriche dettagliate per edges
        neural_scores = []
        embedding_sims = []
        bridge_qualities = []
        
        for u, v, data in subgraph.edges(data=True):
            # Raccogli score se presenti
            if 'neural_score' in data:
                neural_scores.append(data['neural_score'])
            if 'embedding_sim' in data:
                embedding_sims.append(data['embedding_sim'])
            if 'bridge_quality' in data:
                bridge_qualities.append(data['bridge_quality'])
            
            analysis['connections'].append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0),
                'neural_score': data.get('neural_score', 0),
                'embedding_sim': data.get('embedding_sim', 0),
                'source_name': self.equation_map.get(u, {}).get('name', u),
                'target_name': self.equation_map.get(v, {}).get('name', v)
            })
        
        # Calcola metriche aggregate
        analysis['cluster_metrics'] = {
            'avg_neural_score': np.mean(neural_scores) if neural_scores else 0,
            'avg_embedding_sim': np.mean(embedding_sims) if embedding_sims else 0,
            'avg_bridge_quality': np.mean(bridge_qualities) if bridge_qualities else 0,
            'has_gnn_predictions': len(neural_scores) > 0,
            'has_bridges': len(bridge_qualities) > 0
        }
        
        # Info equazioni
        for node in nodes:
            if node in self.equation_map:
                eq = self.equation_map[node]
                analysis['equations'].append({
                    'id': node,
                    'name': eq.get('name', ''),
                    'branch': eq.get('branch', 'Unknown'),
                    'equation': eq.get('equation', ''),
                    'variables': eq.get('variables', [])
                })
                
                # Conta branche
                analysis['branches'][eq.get('branch', 'Unknown')] += 1
                
                # Conta variabili
                for var in eq.get('variables', []):
                    analysis['shared_concepts'][var] += 1
        
        # Filtra concetti condivisi (almeno 2 occorrenze)
        analysis['shared_concepts'] = {k: v for k, v in analysis['shared_concepts'].items() if v >= 2}
        
        # Interpretazione fisica
        analysis['physical_interpretation'] = self._generate_interpretation(analysis)
        
        return analysis
    
    def _generate_interpretation(self, analysis):
        """Genera interpretazione fisica del cluster"""
        interp = []
        
        # Dimensione
        size = analysis['size']
        if size >= 20:
            interp.append(f"🔥 MEGA-CLUSTER: {size} equations")
        elif size >= 10:
            interp.append(f"⭐ Large cluster: {size} equations")
        else:
            interp.append(f"Cluster: {size} equations")
        
        # Densità
        density = analysis['density']
        if density > 0.8:
            interp.append("Very dense (>80% connected)")
        elif density > 0.5:
            interp.append("Dense (>50% connected)")
        
        # Metriche GNN se presenti
        metrics = analysis['cluster_metrics']
        if metrics['has_gnn_predictions']:
            avg_neural = metrics['avg_neural_score']
            if avg_neural > 0.8:
                interp.append(f"GNN confidence: VERY HIGH ({avg_neural:.2f})")
            elif avg_neural > 0.6:
                interp.append(f"GNN confidence: High ({avg_neural:.2f})")
        
        # Cross-branch
        if len(analysis['branches']) > 1:
            interp.append(f"Cross-branch: {', '.join(list(analysis['branches'].keys())[:3])}")
        
        return ' | '.join(interp)
    
    def generate_reports(self, clusters, top_n: int = 100):
        """Genera report enriched"""
        print(f"\n📝 Generating enriched reports...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"cluster_discoveries_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # Combina tutti i cluster
        all_clusters = []
        for cluster_type, cluster_list in clusters.items():
            all_clusters.extend(cluster_list)
        
        # Ordina per dimensione
        all_clusters.sort(key=lambda x: x['size'], reverse=True)
        
        # Report principale
        report = {
            'timestamp': timestamp,
            'total_clusters': len(all_clusters),
            'largest_cluster': all_clusters[0]['size'] if all_clusters else 0,
            'clusters': []
        }
        
        # Analizza top cluster
        for i, cluster in enumerate(all_clusters[:top_n]):
            print(f"\r   Analyzing cluster {i+1}/{min(top_n, len(all_clusters))}...", end="")
            
            analysis = self.analyze_cluster(cluster)
            report['clusters'].append(analysis)
            
            # Salva report individuale per cluster grandi
            if cluster['size'] >= 5:
                individual_file = output_dir / f"cluster_{i+1:03d}_size{cluster['size']}.json"
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)
                
                # Genera visualizzazione
                try:
                    self.visualize_cluster(analysis, i+1, output_dir)
                except:
                    pass
        
        # Salva report principale
        main_file = output_dir / "MAIN_REPORT.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Report testuale
        self.write_text_report(report, output_dir)
        
        print(f"\n✅ Reports saved in {output_dir}/")
        
        return report
    
    def visualize_cluster(self, analysis, idx, output_dir):
        """Visualizza un cluster con labels sempre visibili"""
        nodes = [eq['id'] for eq in analysis['equations']]
        if len(nodes) < 2:
            return
            
        subgraph = self.G.subgraph(nodes)
        
        # Figura più grande per cluster grandi
        fig_size = (12, 8) if len(nodes) <= 15 else (16, 12) if len(nodes) <= 30 else (20, 16)
        plt.figure(figsize=fig_size)
        
        # Layout
        if len(nodes) <= 20:
            pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(subgraph)
        
        # Colori per branca
        colors = []
        for node in nodes:
            branch = self.equation_map.get(node, {}).get('branch', 'Unknown')
            colors.append(BRANCH_COLORS.get(branch, '#808080'))
        
        # Dimensione nodi basata su degree
        node_sizes = [300 + 50 * subgraph.degree(n) for n in nodes]
        
        # Disegna edges colorati per tipo
        edge_colors = []
        edge_widths = []
        for u, v, data in subgraph.edges(data=True):
            if 'neural_score' in data and data['neural_score'] > 0.7:
                edge_colors.append('red')
                edge_widths.append(2.0)
            elif 'bridge_quality' in data and data['bridge_quality'] > 0.5:
                edge_colors.append('blue')
                edge_widths.append(1.5)
            else:
                edge_colors.append('gray')
                edge_widths.append(0.8)
        
        nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, 
                              width=edge_widths, alpha=0.3)
        
        # Disegna nodi
        nx.draw_networkx_nodes(subgraph, pos, node_color=colors, 
                              node_size=node_sizes, alpha=0.9,
                              edgecolors='black', linewidths=1)
        
        # LABELS SEMPRE VISIBILI - strategia adattiva
        labels = {}
        for node in nodes:
            name = self.equation_map.get(node, {}).get('name', node)
            
            # Strategia per abbreviare in base alla dimensione del cluster
            if len(nodes) <= 10:
                # Cluster piccoli: nome completo
                labels[node] = name
            elif len(nodes) <= 20:
                # Cluster medi: abbrevia a 40 caratteri
                if len(name) > 40:
                    labels[node] = name[:37] + '...'
                else:
                    labels[node] = name
            elif len(nodes) <= 40:
                # Cluster grandi: abbrevia a 25 caratteri
                if len(name) > 25:
                    labels[node] = name[:22] + '...'
                else:
                    labels[node] = name
            else:
                # Mega cluster: mostra ID + nome breve
                short_name = name.split(' - ')[0] if ' - ' in name else name[:15]
                labels[node] = f"{node}\n{short_name}"
        
        # Font size adattivo
        if len(nodes) <= 10:
            font_size = 10
        elif len(nodes) <= 20:
            font_size = 8
        elif len(nodes) <= 40:
            font_size = 6
        else:
            font_size = 5
        
        # Disegna labels
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=font_size,
                               font_weight='bold', font_color='darkblue')
        
        # Titolo informativo con metriche
        branches_str = ', '.join([f"{b}({c})" for b, c in list(analysis['branches'].items())[:3]])
        if len(analysis['branches']) > 3:
            branches_str += f" +{len(analysis['branches'])-3} more"
        
        metrics = analysis.get('cluster_metrics', {})
        title_lines = [
            f'Cluster #{idx}: {len(nodes)} equations | Type: {analysis["type"]} | Density: {analysis["density"]:.3f}',
            f'Branches: {branches_str}'
        ]
        if metrics.get('has_gnn_predictions'):
            title_lines.append(f'GNN Score: {metrics["avg_neural_score"]:.3f} | Embedding: {metrics["avg_embedding_sim"]:.3f}')
            
        plt.title('\n'.join(title_lines), fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Aggiungi legenda per le branche
        if len(analysis['branches']) > 1:
            legend_elements = []
            for branch in analysis['branches'].keys():
                if branch in BRANCH_COLORS:
                    legend_elements.append(plt.scatter([], [], c=BRANCH_COLORS[branch], 
                                                      s=100, label=branch))
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper left', 
                          bbox_to_anchor=(1, 1), fontsize=8)
        
        # Salva con DPI più alto per cluster grandi
        dpi = 150 if len(nodes) <= 20 else 200 if len(nodes) <= 40 else 300
        plt.tight_layout()
        plt.savefig(output_dir / f'cluster_{idx:03d}.png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Per cluster molto grandi, genera anche versione ad alta risoluzione
        if len(nodes) >= 30:
            self._generate_high_res_version(analysis, idx, output_dir, subgraph, pos)
    
    def _generate_high_res_version(self, analysis, idx, output_dir, subgraph, pos):
        """Genera versione ad altissima risoluzione per mega cluster"""
        nodes = [eq['id'] for eq in analysis['equations']]
        
        plt.figure(figsize=(30, 30))
        
        # Colori
        colors = []
        for node in nodes:
            branch = self.equation_map.get(node, {}).get('branch', 'Unknown')
            colors.append(BRANCH_COLORS.get(branch, '#808080'))
        
        # Disegna con più dettagli
        nx.draw_networkx_edges(subgraph, pos, alpha=0.15, width=0.5, edge_color='gray')
        nx.draw_networkx_nodes(subgraph, pos, node_color=colors, 
                              node_size=800, alpha=0.95,
                              edgecolors='black', linewidths=2)
        
        # Labels completi con ID
        labels = {}
        for node in nodes:
            name = self.equation_map.get(node, {}).get('name', node)
            labels[node] = f"[{node}]\n{name}"
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8,
                               font_weight='bold', font_color='black',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.7))
        
        plt.title(f'MEGA CLUSTER #{idx} - HIGH RESOLUTION\n'
                 f'{len(nodes)} Interconnected Equations',
                 fontsize=20, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'cluster_{idx:03d}_HD.png', dpi=300, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def write_text_report(self, report, output_dir):
        """Scrive report testuale"""
        with open(output_dir / "REPORT.txt", 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLUSTER DISCOVERY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Total clusters: {report['total_clusters']}\n")
            f.write(f"Largest cluster: {report['largest_cluster']} equations\n\n")
            
            # Top 10 cluster
            for i, cluster in enumerate(report['clusters'][:10], 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"CLUSTER #{i}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Size: {cluster['size']} equations\n")
                f.write(f"Type: {cluster['type']}\n")
                f.write(f"Density: {cluster['density']:.3f}\n")
                
                # Metriche se presenti
                metrics = cluster.get('cluster_metrics', {})
                if metrics.get('has_gnn_predictions'):
                    f.write(f"GNN Score: {metrics['avg_neural_score']:.3f}\n")
                    f.write(f"Embedding Similarity: {metrics['avg_embedding_sim']:.3f}\n")
                
                # Interpretazione
                if 'physical_interpretation' in cluster:
                    f.write(f"\nInterpretation: {cluster['physical_interpretation']}\n")
                
                # Branche
                f.write("\nBranches:\n")
                for branch, count in cluster['branches'].items():
                    f.write(f"  - {branch}: {count}\n")
                
                # Concetti condivisi
                if cluster['shared_concepts']:
                    f.write("\nShared concepts:\n")
                    for concept, count in list(cluster['shared_concepts'].items())[:5]:
                        f.write(f"  - {concept}: {count} times\n")
                
                # Equazioni
                f.write("\nEquations:\n")
                for eq in cluster['equations'][:10]:
                    f.write(f"  • [{eq['id']}] {eq['name']} ({eq['branch']})\n")
                
                if cluster['size'] >= 20:
                    f.write(f"\n🚨 MEGA-CLUSTER ALERT! {cluster['size']} equations!\n")


def main():
    """Esecuzione principale"""
    print("🚀 Starting MEGA CLUSTER DISCOVERY!\n")
    
    # Inizializza
    discovery = MegaClusterDiscovery()
    
    # Trova cluster
    clusters = discovery.find_clusters(min_size=3)
    
    # Genera report
    report = discovery.generate_reports(clusters, top_n=100)
    
    # Summary
    total = sum(len(v) for v in clusters.values())
    print(f"\n{'='*60}")
    print(f"SUMMARY: Found {total} clusters")
    
    if report['largest_cluster'] >= 20:
        print(f"🎉 DISCOVERY OF THE CENTURY! Cluster with {report['largest_cluster']} equations!")
    
    print("✨ Check cluster_discoveries_* folder for details!")


if __name__ == "__main__":
    main()