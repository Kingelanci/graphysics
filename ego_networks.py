#!/usr/bin/env python3
"""
PHYSICS GRAPH VISUALIZER - COMPLETELY FIXED VERSION
===================================================

FIXES:
1. ✅ Cerca file DIRETTAMENTE in /content (non sottocartelle)
2. ✅ 3 LIVELLI MOST CONNECTED: HIGH/MEDIUM/LOW
3. ✅ Layout simmetrici e leggibili
4. ✅ Codice PULITO senza casini
5. ✅ SYNTAX ERROR COMPLETAMENTE RISOLTO
"""

import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif'
})

# Colori branch ORIGINALI
BRANCH_COLORS = {
    'Electromagnetism': '#1f77b4',           
    'Quantum Mechanics': '#ff7f0e',          
    'Statistical Mechanics': '#2ca02c',      
    'Relativity': '#d62728',                 
    'Thermodynamics': '#9467bd',             
    'Atomic and Nuclear Physics': '#8c564b', 
    'Fluid Mechanics': '#e377c2',            
    'Classical Mechanics': '#7f7f7f',        
    'Optics': '#bcbd22',                     
    'Unknown': '#17becf'                     
}

class FixedPhysicsVisualizer:
    
    def __init__(self):
        print("🔥 FIXED PHYSICS VISUALIZER")
        print("=" * 50)
        
        self._load_data_FIXED()
        print(f"✅ Loaded: {len(self.equations)} equations, {len(self.significant_bridges)} bridges")
    
    def _load_data_FIXED(self):
        """Load data - CERCA NEI POSTI GIUSTI!"""
        
        # CERCA DIRETTAMENTE IN /content PRIMA!
        search_paths = [
            Path('/content'),
            Path('.'),
            Path('knowledge_graph_outputs'),
            Path('/content/knowledge_graph_outputs')
        ]
        
        graph_file = None
        for search_path in search_paths:
            if search_path.exists():
                graph_files = list(search_path.glob('physics_knowledge_graph_*.pt'))
                if graph_files:
                    graph_file = max(graph_files, key=lambda p: p.stat().st_mtime)
                    print(f"📁 Found graph: {graph_file}")
                    break
        
        if not graph_file:
            raise FileNotFoundError("❌ NO GRAPH FILES FOUND!")
        
        # Estrai timestamp dal nome file
        timestamp = '_'.join(graph_file.stem.split('_')[-2:])
        base_path = graph_file.parent
        
        # Load metadata
        metadata_file = base_path / f'graph_metadata_{timestamp}.json'
        if not metadata_file.exists():
            # Prova senza timestamp
            metadata_files = list(base_path.glob('graph_metadata_*.json'))
            if metadata_files:
                metadata_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
                print(f"📋 Found metadata: {metadata_file}")
            else:
                raise FileNotFoundError(f"❌ Metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load bridges
        bridge_file = base_path / f'equation_bridges_{timestamp}.json'
        if not bridge_file.exists():
            # Prova senza timestamp
            bridge_files = list(base_path.glob('equation_bridges_*.json'))
            if bridge_files:
                bridge_file = max(bridge_files, key=lambda p: p.stat().st_mtime)
                print(f"🌉 Found bridges: {bridge_file}")
            else:
                raise FileNotFoundError(f"❌ Bridge file not found: {bridge_file}")
        
        with open(bridge_file, 'r') as f:
            self.bridge_data = json.load(f)
        
        # Extract equations
        self.equations = {}
        for node_id, features in self.metadata['node_features'].items():
            if features['node_type'] == 'equation':
                self.equations[node_id] = features
        
        self.significant_bridges = self.bridge_data['significant_bridges']
        
        print(f"✅ Data loaded successfully!")
    
    def clean_variable(self, var):
        """Clean variables - COMPLETELY REWRITTEN TO AVOID SYNTAX ERRORS."""
        import re
        
        # Remove suffixes
        var = re.sub(r'_charge$', '', var)
        var = re.sub(r'_voltage$', '', var)  
        var = re.sub(r'_mass$', '', var)
        var = re.sub(r'_force$', '', var)
        var = re.sub(r'_energy$', '', var)
        var = re.sub(r'_field$', '', var)
        var = re.sub(r'_test$', '', var)
        var = re.sub(r'_[0-9]+$', '', var)
        
        # Remove prefixes
        var = re.sub(r'^q_', '', var)
        var = re.sub(r'^k_', '', var)
        var = re.sub(r'^m_', '', var)
        var = re.sub(r'^E_', '', var)
        
        # Substitutions
        substitutions = {
            'epsilon_0': 'ε₀', 
            'mu_0': 'μ₀', 
            'k_e': 'k',
            'planck': 'h', 
            'boltzmann': 'k', 
            'light': 'c',
            'q_charge': 'q', 
            'F_force': 'F'
        }
        
        if var in substitutions:
            return substitutions[var]
        
        if len(var) <= 3 and var.isalpha():
            return var
        
        return None
    
    def create_most_connected_networks(self, output_dir='/content'):
        """CREA 3 LIVELLI MOST CONNECTED: HIGH/MEDIUM/LOW"""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n🎯 Creating 3-LEVEL MOST CONNECTED networks...")
        
        # Calcola grado di connettività per ogni equazione
        eq_degrees = {}
        for bridge in self.significant_bridges:
            eq_degrees[bridge['source']] = eq_degrees.get(bridge['source'], 0) + 1
            eq_degrees[bridge['target']] = eq_degrees.get(bridge['target'], 0) + 1
        
        # Ordina per grado di connettività
        sorted_equations = sorted(eq_degrees.items(), key=lambda x: x[1], reverse=True)
        
        print(f"📊 Total equations with connections: {len(sorted_equations)}")
        
        # DEFINISCI 3 LIVELLI
        levels = {
            'HIGH': sorted_equations[:30],      # Top 30 più connesse
            'MEDIUM': sorted_equations[30:70],  # 40 medie
            'LOW': sorted_equations[70:120]     # 50 basse
        }
        
        colors = {'HIGH': '#d62728', 'MEDIUM': '#ff7f0e', 'LOW': '#2ca02c'}
        
        for level_name, equations in levels.items():
            if not equations:
                continue
                
            print(f"\n🔴 Creating {level_name} connectivity network ({len(equations)} equations)...")
            
            # Estrai IDs delle equazioni per questo livello
            eq_ids = set([eq_id for eq_id, degree in equations])
            
            # Trova bridges tra queste equazioni
            level_bridges = []
            for bridge in self.significant_bridges:
                if bridge['source'] in eq_ids and bridge['target'] in eq_ids:
                    level_bridges.append(bridge)
            
            print(f"   📊 {len(level_bridges)} bridges in {level_name} level")
            
            if len(level_bridges) == 0:
                continue
            
            # Crea network
            G = nx.Graph()
            
            # Add nodes
            for eq_id in eq_ids:
                if eq_id in self.equations:
                    eq_data = self.equations[eq_id]
                    G.add_node(eq_id, 
                              name=eq_data['name'],
                              branch=eq_data['branch'],
                              equation=eq_data.get('equation', ''),
                              degree=eq_degrees[eq_id])
            
            # Add edges
            for bridge in level_bridges:
                if bridge['source'] in G and bridge['target'] in G:
                    G.add_edge(bridge['source'], bridge['target'], 
                              weight=bridge['weight'])
            
            print(f"   📈 Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # LAYOUT SIMMETRICO BASATO SU DIMENSIONE
            pos = self._get_symmetric_layout(G)
            
            # CREA VISUALIZZAZIONE
            self._create_level_visualization(G, pos, level_name, colors[level_name], 
                                           equations, output_dir)
    
    def _get_symmetric_layout(self, G):
        """Layout simmetrico basato su dimensione del grafo."""
        n_nodes = G.number_of_nodes()
        
        if n_nodes <= 20:
            # Piccoli: layout circolare perfetto
            print(f"   🔄 Using CIRCULAR layout ({n_nodes} nodes)")
            return nx.circular_layout(G)
        elif n_nodes <= 40:
            # Medi: layout shell (cerchi concentrici)
            print(f"   🔄 Using SHELL layout ({n_nodes} nodes)")
            return nx.shell_layout(G)
        else:
            # Grandi: layout force-directed simmetrico
            print(f"   🔄 Using FORCE-DIRECTED layout ({n_nodes} nodes)")
            try:
                return nx.kamada_kawai_layout(G)
            except:
                return nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    def _create_level_visualization(self, G, pos, level_name, level_color, equations, output_dir):
        """Crea visualizzazione per un livello."""
        
        # FIGURA
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw edges con peso
        for (u, v, data) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            weight = data['weight']
            
            # Spessore basato su peso
            width = 0.5 + 2 * (weight - 0.5) if weight > 0.5 else 0.5
            alpha = 0.4 + 0.4 * weight if weight < 1 else 0.8
            
            ax.plot([x1, x2], [y1, y2], color='gray', alpha=alpha, linewidth=width)
        
        # Draw nodes COLORATI PER BRANCH
        for node in G.nodes():
            x, y = pos[node]
            branch = G.nodes[node]['branch']
            degree = G.nodes[node]['degree']
            
            # Colore per branch
            color = BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown'])
            
            # Dimensione basata su grado
            size = 100 + degree * 20
            
            # HIGHLIGHT per questo livello
            edge_color = level_color
            edge_width = 3
            
            ax.scatter(x, y, s=size, c=color, edgecolors=edge_color, 
                      linewidth=edge_width, alpha=0.9, zorder=2)
        
        # Labels per nodi importanti
        degree_threshold = np.percentile([G.nodes[n]['degree'] for n in G.nodes()], 70)
        
        for node in G.nodes():
            if G.nodes[node]['degree'] >= degree_threshold:
                x, y = pos[node]
                name = G.nodes[node]['name']
                short_name = name.split(' - ')[0][:15]
                
                ax.annotate(short_name, (x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # TITLE
        min_degree = min([degree for eq_id, degree in equations])
        max_degree = max([degree for eq_id, degree in equations])
        
        ax.set_title(f'{level_name} Connectivity Physics Network\n' +
                    f'{G.number_of_nodes()} Equations • {G.number_of_edges()} Bridges • ' +
                    f'Connections: {min_degree}-{max_degree}',
                    fontsize=18, fontweight='bold', pad=20, color=level_color)
        
        # STATS BOX
        stats_text = f"{level_name} CONNECTIVITY\n"
        stats_text += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        stats_text += f"Equations: {G.number_of_nodes()}\n"
        stats_text += f"Bridges: {G.number_of_edges()}\n"
        stats_text += f"Density: {nx.density(G):.3f}\n"
        stats_text += f"Avg Degree: {2*G.number_of_edges()/G.number_of_nodes():.1f}\n"
        stats_text += f"Connection Range: {min_degree}-{max_degree}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor=level_color, alpha=0.2),
               verticalalignment='top', fontsize=11, fontweight='bold')
        
        # BRANCH LEGEND
        branches_present = set(G.nodes[n]['branch'] for n in G.nodes())
        legend_elements = []
        for branch in sorted(branches_present):
            color = BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown'])
            count = sum(1 for n in G.nodes() if G.nodes[n]['branch'] == branch)
            legend_elements.append(mpatches.Patch(color=color, label=f'{branch} ({count})'))
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1), frameon=True, fontsize=9)
        
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        
        # SAVE
        output_file = Path(output_dir) / f'physics_bridges_{level_name}_CONNECTIVITY.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✅ Saved: {output_file}")
    
    def create_ego_TUTTI_COLORATI(self, output_dir='/content'):
        """EGO networks con TUTTI i concetti e TUTTE le equazioni COLORATE."""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n🎨 Creating EGO TUTTI COLORATI...")
        
        # Extract concepts from equations
        concepts = defaultdict(set)
        for eq_id, eq_data in self.equations.items():
            for var in eq_data.get('variables', []):
                clean_var = self.clean_variable(var)
                if clean_var:
                    concepts[clean_var].add(eq_id)
        
        # TUTTI i concetti importanti!
        concept_counts = [(c, len(eqs)) for c, eqs in concepts.items()]
        top_concepts = sorted(concept_counts, key=lambda x: x[1], reverse=True)[:12]
        
        print("Top 12 concepts TUTTI:")
        for concept, count in top_concepts:
            print(f"   {concept}: {count} equations")
        
        # Create ego networks TUTTI
        for i, (concept, eq_count) in enumerate(top_concepts):
            if eq_count < 5:
                continue
            
            print(f"  🎨 Creating ego for {concept} with ALL {eq_count} equations...")
            
            # TUTTE le equazioni!
            concept_eqs = list(concepts[concept])
            
            # Create ego network
            G = nx.Graph()
            concept_node = f"concept_{concept}"
            G.add_node(concept_node, node_type='concept', name=concept)
            
            # Add ALL equations
            for eq_id in concept_eqs:
                if eq_id in self.equations:
                    eq_data = self.equations[eq_id]
                    G.add_node(eq_id, 
                              branch=eq_data['branch'],
                              name=eq_data['name'],
                              equation=eq_data.get('equation', ''))
                    G.add_edge(concept_node, eq_id)
            
            # LAYOUT RADIALE adattivo
            pos = {}
            pos[concept_node] = (0, 0)
            
            other_nodes = [n for n in G.nodes() if n != concept_node]
            n_nodes = len(other_nodes)
            
            if n_nodes > 0:
                angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
                # Radius adattivo
                if n_nodes <= 10:
                    radius = 1.0
                elif n_nodes <= 20:
                    radius = 1.3
                elif n_nodes <= 30:
                    radius = 1.6
                else:
                    radius = 2.0
                
                for j, node in enumerate(other_nodes):
                    pos[node] = (radius * np.cos(angles[j]), radius * np.sin(angles[j]))
            
            # FIGURA
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Golden rays
            for u, v in G.edges():
                if concept_node in [u, v]:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], 'gold', alpha=0.6, linewidth=1)
            
            # CONCEPT NODE - CERCHIO DORATO
            center_x, center_y = pos[concept_node]
            
            concept_circle = plt.Circle((center_x, center_y), 0.15, 
                                      color='gold', alpha=0.9, 
                                      edgecolor='darkorange', linewidth=4)
            ax.add_patch(concept_circle)
            
            ax.text(center_x, center_y, concept.upper(), 
                   fontsize=20, fontweight='bold',
                   ha='center', va='center', color='darkred')
            
            # EQUATION RECTANGLES - TUTTI COLORATI!
            for node in other_nodes:
                eq_data = G.nodes[node]
                branch = eq_data['branch']
                name = eq_data['name']
                equation = eq_data.get('equation', '')
                
                x, y = pos[node]
                
                # COLORE PER BRANCH!
                color = BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown'])
                
                # Rettangoli adattivi
                if n_nodes <= 15:
                    width = 0.5
                    height = 0.3
                    font_name = 9
                    font_eq = 8
                elif n_nodes <= 25:
                    width = 0.4
                    height = 0.25
                    font_name = 8
                    font_eq = 7
                else:
                    width = 0.35
                    height = 0.2
                    font_name = 7
                    font_eq = 6
                
                rect = mpatches.FancyBboxPatch(
                    (x - width/2, y - height/2), width, height,
                    boxstyle="round,pad=0.03",
                    facecolor=color,  # COLORI BRANCH!
                    edgecolor='black',
                    alpha=0.85,
                    linewidth=1.5
                )
                ax.add_patch(rect)
                
                # TESTO LEGGIBILE
                name_parts = name.split(' - ')
                short_name = name_parts[0]
                if len(short_name) > 15:
                    short_name = short_name[:12] + "..."
                
                ax.text(x, y + height/3, short_name,
                       fontsize=font_name, fontweight='bold', 
                       ha='center', va='center', color='white')
                
                # Equation formula
                if equation and len(equation) < 25:
                    eq_short = equation[:20]
                    ax.text(x, y - height/3, eq_short,
                           fontsize=font_eq, style='italic',
                           ha='center', va='center', color='white')
            
            # Title
            ax.set_title(f'Physics Concept: "{concept.upper()}" Domain\n' + 
                        f'({eq_count} equations total)',
                        fontsize=16, fontweight='bold', pad=20)
            
            # LEGEND COLORATA!
            branches_present = set(G.nodes[n]['branch'] for n in other_nodes)
            legend_elements = []
            for branch in sorted(branches_present):
                color = BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown'])
                count = sum(1 for n in other_nodes if G.nodes[n]['branch'] == branch)
                legend_elements.append(mpatches.Patch(color=color, label=f'{branch[:15]} ({count})'))
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1, 1), frameon=True, fontsize=9)
            
            # Limits adattivi
            max_radius = 2.0 if n_nodes > 30 else (1.6 if n_nodes > 20 else 1.3)
            limit = max_radius + 0.8
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect('equal')
            ax.axis('off')
            
            plt.tight_layout()
            
            output_file = Path(output_dir) / f'ego_TUTTI_COLORATI_{i+1:02d}_{concept}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    ✅ Ego COLORATO saved: {output_file}")
    
    def create_lorenz_NETWORK_COMPATTO(self, output_dir='/content'):
        """LORENZ NETWORK COMPATTO - Stile GRAFO con nodi colorati e testo su 2 righe!"""
        
        # Find most connected equation (Lorenz)
        eq_degrees = {}
        for bridge in self.significant_bridges:
            eq_degrees[bridge['source']] = eq_degrees.get(bridge['source'], 0) + 1
            eq_degrees[bridge['target']] = eq_degrees.get(bridge['target'], 0) + 1
        
        top_eq = max(eq_degrees.items(), key=lambda x: x[1])
        eq_id, degree = top_eq
        
        print(f"\n🌟 Creating LORENZ NETWORK COMPATTO for: {eq_id} ({degree} connections)")
        
        eq_data = self.equations[eq_id]
        
        # Get connected equations
        connected_eqs = set()
        bridge_weights = {}
        
        for bridge in self.significant_bridges:
            # Exclude Lorenz-Lorenz connections
            source_name = self.equations.get(bridge['source'], {}).get('name', '')
            target_name = self.equations.get(bridge['target'], {}).get('name', '')
            
            if (('Lorenz' in source_name or 'Lorentz' in source_name) and 
                ('Lorenz' in target_name or 'Lorentz' in target_name)):
                continue
            
            if bridge['source'] == eq_id:
                connected_eqs.add(bridge['target'])
                bridge_weights[(eq_id, bridge['target'])] = bridge['weight']
            elif bridge['target'] == eq_id:
                connected_eqs.add(bridge['source'])
                bridge_weights[(bridge['source'], eq_id)] = bridge['weight']
        
        # Take top 20 connected equations
        connected_list = list(connected_eqs)
        if len(connected_list) > 20:
            weighted_connections = []
            for conn_eq in connected_list:
                weight = bridge_weights.get((eq_id, conn_eq), bridge_weights.get((conn_eq, eq_id), 0))
                weighted_connections.append((conn_eq, weight))
            
            weighted_connections.sort(key=lambda x: x[1], reverse=True)
            connected_list = [eq for eq, _ in weighted_connections[:20]]
        
        # Create network
        G = nx.Graph()
        
        # Add central equation
        G.add_node(eq_id, **eq_data, is_central=True)
        
        # Add connected equations
        for conn_eq in connected_list:
            if conn_eq in self.equations:
                G.add_node(conn_eq, **self.equations[conn_eq], is_central=False)
                weight = bridge_weights.get((eq_id, conn_eq), bridge_weights.get((conn_eq, eq_id), 0.5))
                G.add_edge(eq_id, conn_eq, weight=weight)
        
        # Add inter-connections between non-central equations
        for i, eq1 in enumerate(connected_list):
            for j, eq2 in enumerate(connected_list):
                if i >= j:
                    continue
                    
                for bridge in self.significant_bridges:
                    if ((bridge['source'] == eq1 and bridge['target'] == eq2) or 
                        (bridge['source'] == eq2 and bridge['target'] == eq1)):
                        
                        eq1_name = self.equations.get(eq1, {}).get('name', '')
                        eq2_name = self.equations.get(eq2, {}).get('name', '')
                        
                        if (('Lorenz' in eq1_name or 'Lorentz' in eq1_name) and 
                            ('Lorenz' in eq2_name or 'Lorentz' in eq2_name)):
                            break
                        
                        if bridge['weight'] > 0.6:
                            G.add_edge(eq1, eq2, weight=bridge['weight'])
                        break
        
        print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # LAYOUT GRAFO PROFESSIONALE (non EGO style)
        n_nodes = G.number_of_nodes()
        
        try:
            if n_nodes <= 15:
                # Small networks: circular layout
                pos = nx.circular_layout(G)
                print(f"   🔄 Using CIRCULAR layout ({n_nodes} nodes)")
            elif n_nodes <= 25:
                # Medium networks: shell layout with central node in center
                shells = [[eq_id], [n for n in G.nodes() if n != eq_id]]
                pos = nx.shell_layout(G, nlist=shells)
                print(f"   🔄 Using SHELL layout ({n_nodes} nodes)")
            else:
                # Large networks: force-directed layout
                pos = nx.kamada_kawai_layout(G)
                print(f"   🔄 Using KAMADA-KAWAI layout ({n_nodes} nodes)")
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
            print(f"   🔄 Using SPRING layout (fallback)")
        
        # FIGURA STILE GRAFO PROFESSIONALE
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Draw edges con peso e colore basato su strength
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            weight = G[u][v]['weight']
            
            # Edge styling basato su peso
            if weight > 0.9:
                color = '#d62728'  # Red per strongest
                alpha = 0.8
                linewidth = 3
            elif weight > 0.8:
                color = '#ff7f0e'  # Orange per strong
                alpha = 0.7
                linewidth = 2.5
            elif weight > 0.7:
                color = '#2ca02c'  # Green per medium
                alpha = 0.6
                linewidth = 2
            else:
                color = '#7f7f7f'  # Gray per weak
                alpha = 0.4
                linewidth = 1
            
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth, zorder=1)
        
        # Draw nodes come CERCHI COLORATI (stile network graph)
        for node in G.nodes():
            node_data = G.nodes[node]
            branch = node_data['branch']
            name = node_data['name']
            equation = node_data.get('equation', '')
            is_central = node_data.get('is_central', False)
            
            x, y = pos[node]
            
            # COLORE PER BRANCH
            color = BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown'])
            
            # Dimensioni basate su centralità
            if is_central:
                node_size = 1200  # Nodo centrale più grande
                edge_color = '#d62728'  # Rosso per centrale
                edge_width = 4
                alpha = 1.0
            else:
                node_size = 800   # Nodi normali
                edge_color = 'black'
                edge_width = 2
                alpha = 0.9
            
            # Draw NODE come cerchio
            ax.scatter(x, y, s=node_size, c=color, edgecolors=edge_color, 
                      linewidth=edge_width, alpha=alpha, zorder=3)
        
        # Add LABELS su 2 righe per leggibilità migliore
        for node in G.nodes():
            node_data = G.nodes[node]
            name = node_data['name']
            equation = node_data.get('equation', '')
            is_central = node_data.get('is_central', False)
            
            x, y = pos[node]
            
            # FORMATTA NOME SU 2 RIGHE se necessario
            name_parts = name.split(' - ')
            line1 = name_parts[0]
            
            # Se il nome è troppo lungo, spezzalo
            if len(line1) > 15:
                words = line1.split()
                if len(words) > 1:
                    mid = len(words) // 2
                    line1 = ' '.join(words[:mid])
                    line2 = ' '.join(words[mid:])
                else:
                    line1 = line1[:12]
                    line2 = line1[12:] if len(line1) > 12 else ""
            else:
                line2 = name_parts[1] if len(name_parts) > 1 else ""
            
            # Limita lunghezza righe
            if len(line1) > 15:
                line1 = line1[:12] + "..."
            if len(line2) > 15:
                line2 = line2[:12] + "..."
            
            # Font size basato su centralità
            if is_central:
                font_size = 11
                font_weight = 'bold'
                text_color = 'white'
                bbox_color = 'darkred'
                bbox_alpha = 0.9
            else:
                font_size = 9
                font_weight = 'bold'
                text_color = 'black'
                bbox_color = 'white'
                bbox_alpha = 0.8
            
            # TESTO SU 2 RIGHE
            if line2:
                # Due righe
                ax.text(x, y + 0.05, line1, fontsize=font_size, fontweight=font_weight,
                       ha='center', va='center', color=text_color, zorder=4,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color, 
                                alpha=bbox_alpha, edgecolor='black'))
                ax.text(x, y - 0.05, line2, fontsize=font_size-1, fontweight='normal',
                       ha='center', va='center', color=text_color, zorder=4,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=bbox_color, 
                                alpha=bbox_alpha-0.1, edgecolor='black'))
            else:
                # Una riga
                ax.text(x, y, line1, fontsize=font_size, fontweight=font_weight,
                       ha='center', va='center', color=text_color, zorder=4,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color, 
                                alpha=bbox_alpha, edgecolor='black'))
            
            # Aggiungi FORMULA sotto se disponibile e non troppo lunga
            if equation and len(equation) < 20 and not is_central:
                eq_short = equation[:15] + "..." if len(equation) > 15 else equation
                ax.text(x, y - 0.15, eq_short, fontsize=7, style='italic',
                       ha='center', va='center', color='darkblue', zorder=4,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', 
                                alpha=0.7, edgecolor='none'))
        
        # Title professionale
        central_name = eq_data['name'].split(' - ')[0]
        ax.set_title(f'Physics Network Hub: "{central_name}"\n' + 
                    f'{G.number_of_nodes()} Connected Equations • {G.number_of_edges()} Bridges • Density: {nx.density(G):.3f}',
                    fontsize=18, fontweight='bold', pad=25)
        
        # Stats box dettagliato
        stats_text = f"NETWORK ANALYSIS\n"
        stats_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        stats_text += f"Central Hub: {central_name[:20]}\n"
        stats_text += f"Hub Connections: {degree}\n"
        stats_text += f"Connected Equations: {len(connected_list)}\n"
        stats_text += f"Total Bridges: {G.number_of_edges()}\n"
        stats_text += f"Network Density: {nx.density(G):.3f}\n"
        stats_text += f"Avg Clustering: {nx.average_clustering(G):.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
               verticalalignment='top', fontsize=11, fontweight='bold')
        
        # EDGE WEIGHT LEGEND
        edge_legend_text = f"CONNECTION STRENGTH\n"
        edge_legend_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        edge_legend_text += f"🔴 Very Strong (>0.9)\n"
        edge_legend_text += f"🟠 Strong (0.8-0.9)\n"
        edge_legend_text += f"🟢 Medium (0.7-0.8)\n"
        edge_legend_text += f"⚪ Weak (<0.7)"
        
        ax.text(0.02, 0.35, edge_legend_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9),
               verticalalignment='top', fontsize=10, fontweight='bold')
        
        # PHYSICS BRANCH LEGEND
        branches_present = set(G.nodes[n]['branch'] for n in G.nodes())
        legend_elements = []
        for branch in sorted(branches_present):
            color = BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown'])
            count = sum(1 for n in G.nodes() if G.nodes[n]['branch'] == branch)
            legend_elements.append(mpatches.Patch(color=color, label=f'{branch} ({count})'))
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1), frameon=True, fontsize=10,
                 title='Physics Branches', title_fontsize=12)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Adjust limits per vedere tutto
        x_coords = [pos[node][0] for node in G.nodes()]
        y_coords = [pos[node][1] for node in G.nodes()]
        
        margin = 0.3
        ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        
        plt.tight_layout()
        
        output_file = Path(output_dir) / f'lorenz_NETWORK_COMPATTO_{eq_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ LORENZ NETWORK COMPATTO (Graph style) saved: {output_file}")
    
    def create_complete_network_400(self, output_dir='/content'):
        """CREA GRAFO COMPLESSIVO con TUTTE le 400 equazioni - NODI VISIBILI AL CENTRO!"""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\n🌍 Creating COMPLETE network with ALL {len(self.equations)} equations...")
        
        # Get largest connected component for better visualization
        G_full = nx.Graph()
        
        # Add ALL equations
        for eq_id, eq_data in self.equations.items():
            G_full.add_node(eq_id,
                      name=eq_data['name'],
                      branch=eq_data['branch'],
                      equation=eq_data.get('equation', ''))
        
        # Add ALL bridges
        bridge_count = 0
        for bridge in self.significant_bridges:
            if bridge['source'] in G_full and bridge['target'] in G_full:
                G_full.add_edge(bridge['source'], bridge['target'], 
                          weight=bridge['weight'])
                bridge_count += 1
        
        # Use largest connected component if graph is fragmented
        if not nx.is_connected(G_full):
            components = list(nx.connected_components(G_full))
            largest_comp = max(components, key=len)
            G = G_full.subgraph(largest_comp).copy()
            print(f"✅ Using largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        else:
            G = G_full
            print(f"✅ Using complete graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Layout ottimizzato per grafi grandi
        print("🔄 Computing layout for large network...")
        try:
            pos = nx.fruchterman_reingold_layout(G, k=3.0, iterations=100)
        except:
            try:
                pos = nx.spring_layout(G, k=5.0, iterations=200, seed=42)
            except:
                print("⚠️ Using random layout...")
                np.random.seed(42)
                pos = {node: (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) 
                       for node in G.nodes()}
        
        # FIGURA GIGANTE per 400 equazioni
        fig, ax = plt.subplots(figsize=(24, 20))
        
        # Calcola gradi per dimensioni
        degrees = dict(G.degree())
        max_degree = max(degrees.values())
        min_degree = min(degrees.values())
        
        # Draw edges PRIMA (background)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        min_w, max_w = min(edge_weights), max(edge_weights)
        
        for (u, v), weight in zip(G.edges(), edge_weights):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Thickness and alpha based on weight
            thickness = 0.1 + 0.8 * (weight - min_w) / (max_w - min_w)
            alpha = 0.1 + 0.3 * (weight - min_w) / (max_w - min_w)
            
            ax.plot([x1, x2], [y1, y2], color='gray', alpha=alpha, linewidth=thickness, zorder=1)
        
        # Draw ALL nodes COLORATI PER BRANCH (foreground)
        branch_counts = defaultdict(int)
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            branch = G.nodes[node]['branch']
            degree = degrees[node]
            
            branch_counts[branch] += 1
            node_colors.append(BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown']))
            
            # Size based on degree (connectivity)
            size = 15 + (degree - min_degree) / max(1, (max_degree - min_degree)) * 80
            node_sizes.append(size)
        
        # Draw nodes as scatter plot (più veloce per tanti punti)
        x_coords = [pos[node][0] for node in G.nodes()]
        y_coords = [pos[node][1] for node in G.nodes()]
        
        scatter = ax.scatter(x_coords, y_coords, 
                           c=node_colors, s=node_sizes, 
                           alpha=0.8, edgecolors='black', linewidths=0.5, zorder=2)
        
        # Labels solo per TOP HUBS (most connected equations)
        degree_dict = dict(G.degree())
        top_hubs = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)[:25]
        
        for eq_id in top_hubs:
            x, y = pos[eq_id]
            name = G.nodes[eq_id]['name']
            short_name = name.split(' - ')[0][:12]
            ax.annotate(short_name, (x, y), xytext=(3, 3), textcoords='offset points',
                       fontsize=6, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # PROFESSIONAL TITLE
        ax.set_title(f'Complete Physics Knowledge Graph\n' +
                    f'{G.number_of_nodes()} Equations • {G.number_of_edges()} Bridges • ' +
                    f'{len(branch_counts)} Physics Branches • Density: {nx.density(G):.4f}',
                    fontsize=18, fontweight='bold', pad=30)
        
        # COMPREHENSIVE LEGEND
        legend_elements = []
        sorted_branches = sorted(branch_counts.items(), key=lambda x: x[1], reverse=True)
        
        for branch, count in sorted_branches:
            color = BRANCH_COLORS.get(branch, BRANCH_COLORS['Unknown'])
            percentage = 100 * count / G.number_of_nodes()
            legend_elements.append(mpatches.Patch(color=color, 
                                                 label=f'{branch} ({count}, {percentage:.1f}%)'))
        
        # Split legend in two columns if too many branches
        if len(legend_elements) > 6:
            mid = len(legend_elements) // 2
            legend1 = ax.legend(handles=legend_elements[:mid], 
                               loc='upper left', bbox_to_anchor=(0, 1), 
                               frameon=True, fontsize=9)
            ax.add_artist(legend1)
            ax.legend(handles=legend_elements[mid:], 
                     loc='upper left', bbox_to_anchor=(0, 0.6), 
                     frameon=True, fontsize=9)
        else:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                     frameon=True, fontsize=9)
        
        # STATISTICS BOX
        stats_text = f"COMPLETE PHYSICS NETWORK\n"
        stats_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        stats_text += f"Total Equations: {G.number_of_nodes():,}\n"
        stats_text += f"Total Bridges: {G.number_of_edges():,}\n"
        stats_text += f"Physics Branches: {len(branch_counts)}\n"
        stats_text += f"Network Density: {nx.density(G):.4f}\n"
        stats_text += f"Average Connections: {2*G.number_of_edges()/G.number_of_nodes():.1f}\n"
        stats_text += f"Most Connected: {max_degree} bridges\n"
        stats_text += f"Coverage: {100*G.number_of_nodes()/len(self.equations):.1f}% of database"
        
        ax.text(0.99, 0.01, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=11, fontweight='bold')
        
        # METHODOLOGY BOX
        method_text = f"VISUALIZATION METHOD\n"
        method_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        method_text += f"Layout: Force-directed algorithm\n"
        method_text += f"Node size: ∝ equation connectivity\n"
        method_text += f"Node color: physics branch\n"
        method_text += f"Edge width: ∝ bridge strength\n"
        method_text += f"Labels: top 25 most connected\n"
        method_text += f"Component: largest connected"
        
        ax.text(0.01, 0.01, method_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
               verticalalignment='bottom', horizontalalignment='left',
               fontsize=9, fontweight='bold')
        
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'physics_knowledge_graph_COMPLETE_400.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ COMPLETE NETWORK saved: {output_file}")
        print(f"📊 Final stats: {G.number_of_nodes()} equations, {G.number_of_edges()} bridges")
    
    def run_fixed_analysis(self):
        """RUN ANALISI COMPLETA FIXED con EGO e LORENZ!"""
        print("\n🚀 RUNNING COMPLETE FIXED ANALYSIS!")
        print("=" * 60)
        
        output_dir = '/content'
        
        # 1. 3-LEVEL MOST CONNECTED NETWORKS
        self.create_most_connected_networks(output_dir)
        
        # 2. EGO NETWORKS TUTTI COLORATI
        self.create_ego_TUTTI_COLORATI(output_dir)
        
        # 3. LORENZ NETWORK COMPATTO (stile EGO)
        self.create_lorenz_NETWORK_COMPATTO(output_dir)
        
        # 4. COMPLETE 400 EQUATION NETWORK (con nodi visibili!)
        self.create_complete_network_400(output_dir)
        
        print(f"\n✅ COMPLETE FIXED ANALYSIS DONE!")
        print(f"📁 Files saved in: {output_dir}")
        
        # Lista file creati
        output_path = Path(output_dir)
        figures = list(output_path.glob('physics_*.png')) + list(output_path.glob('ego_*.png')) + list(output_path.glob('lorenz_*.png'))
        print(f"🎯 Created {len(figures)} figures:")
        for fig in sorted(figures):
            print(f"   📄 {fig.name}")
            
        print(f"\n🎨 VISUALIZATION TYPES CREATED:")
        print(f"   🔴 HIGH/MEDIUM/LOW connectivity networks (3 files)")
        print(f"   🌟 EGO networks for top concepts (12 files)")  
        print(f"   ⚡ LORENZ compact network EGO-style (1 file)")
        print(f"   🌍 Complete 400-equation overview (1 file)")
        print(f"   📊 Total: ~17 publication-quality visualizations!")


def main():
    """Main function."""
    try:
        viz = FixedPhysicsVisualizer()
        viz.run_fixed_analysis()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()