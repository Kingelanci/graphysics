#!/usr/bin/env python3
"""
AST ANALYSIS FOR PHYSICS EQUATIONS - FINAL 100% VERSION
======================================================
Debug dettagliato e gestione perfetta dei caratteri greci
"""

import json
import numpy as np
import sympy as sp
import re
from typing import Dict, List, Any, Tuple, Set, Union, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime
from pathlib import Path

# Visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Warning: matplotlib/seaborn not available. Visualizations will be skipped.")

@dataclass
class ASTNode:
    """Rappresentazione di un nodo dell'AST."""
    node_type: str
    value: Union[str, float, None]
    children: List['ASTNode']
    depth: int
    parent: Optional['ASTNode'] = None
    position: int = 0
    
    def __str__(self):
        return f"{self.node_type}({self.value})"
    
    def get_subtree_size(self) -> int:
        """Calcola la dimensione del sottoalbero."""
        return 1 + sum(child.get_subtree_size() for child in self.children)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte il nodo in dizionario serializzabile."""
        return {
            'node_type': self.node_type,
            'value': self.value if isinstance(self.value, (str, int, float, type(None))) else str(self.value),
            'depth': self.depth,
            'position': self.position,
            'children_count': len(self.children)
        }

@dataclass
class AdvancedFeatures:
    """Features avanzate dell'AST."""
    tree_imbalance: float = 0.0
    average_path_length: float = 0.0
    clustering_coefficient: float = 0.0
    common_subpatterns: Dict[str, int] = field(default_factory=dict)
    operator_sequences: List[str] = field(default_factory=list)
    nested_operations: Dict[str, int] = field(default_factory=dict)
    has_transcendental: bool = False
    has_trigonometric: bool = False
    has_logarithmic: bool = False
    has_exponential: bool = False
    polynomial_degree: Optional[int] = None
    is_symmetric: bool = False
    has_repeated_subexpressions: bool = False

class EquationParser:
    """Parser avanzato per convertire espressioni SymPy in AST strutturati."""
    
    TRANSCENDENTAL_FUNCS = {'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt'}
    TRIG_FUNCS = {'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan'}
    LOG_FUNCS = {'log', 'ln', 'log10', 'log2'}
    EXP_FUNCS = {'exp', 'Pow'}
    
    def __init__(self, sympy_expr):
        self.expr = sympy_expr
        self.raw_expr_str = str(sympy_expr)
        self.ast_root = None
        self.structural_features = {}
        self.numerical_components = {}
        self.advanced_features = AdvancedFeatures()
        
        # Costruisci AST
        self.ast_root = self.to_ast()
        if self.ast_root:
            self._compute_all_features()
    
    def to_ast(self) -> Optional[ASTNode]:
        """Attraversa ricorsivamente l'espressione SymPy."""
        try:
            return self._sympy_to_ast_recursive(self.expr, depth=0, parent=None, position=0)
        except Exception as e:
            return None
    
    def _sympy_to_ast_recursive(self, expr, depth: int, parent: Optional[ASTNode], position: int) -> ASTNode:
        """Conversione ricorsiva SymPy -> AST con informazioni aggiuntive."""
        
        # Determina il tipo di nodo
        node_type = type(expr).__name__
        node_value = None
        
        # Gestione casi specifici con conversione sicura
        if expr.is_Atom:
            if expr.is_Symbol:
                node_value = str(expr)
            elif expr.is_Number:
                try:
                    # Converti numeri SymPy in float Python nativi
                    if hasattr(expr, 'evalf'):
                        node_value = float(expr.evalf())
                    else:
                        node_value = float(expr)
                except:
                    node_value = str(expr)
            else:
                node_value = str(expr)
        
        # Crea nodo AST
        ast_node = ASTNode(
            node_type=node_type,
            value=node_value,
            children=[],
            depth=depth,
            parent=parent,
            position=position
        )
        
        # Ricorsione sui figli
        if hasattr(expr, 'args') and expr.args:
            for i, arg in enumerate(expr.args):
                child_node = self._sympy_to_ast_recursive(arg, depth + 1, ast_node, i)
                ast_node.children.append(child_node)
        
        return ast_node
    
    def _compute_all_features(self):
        """Calcola tutte le features dell'AST."""
        self.structural_features = self.get_structural_features()
        self.numerical_components = self.get_numerical_components()
        self._compute_advanced_features()
    
    def _compute_advanced_features(self):
        """Calcola features avanzate."""
        if not self.ast_root:
            return
        
        self.advanced_features.tree_imbalance = self._calculate_tree_imbalance()
        self.advanced_features.average_path_length = self._calculate_avg_path_length()
        self._detect_patterns()
        self._detect_mathematical_types()
        self._detect_symmetry()
    
    def _calculate_tree_imbalance(self) -> float:
        """Calcola l'imbalance dell'albero."""
        def get_height(node: ASTNode) -> int:
            if not node.children:
                return 1
            return 1 + max(get_height(child) for child in node.children)
        
        def calculate_imbalance(node: ASTNode) -> float:
            if not node.children:
                return 0.0
            
            heights = [get_height(child) for child in node.children]
            if len(heights) < 2:
                return 0.0
            
            max_h, min_h = max(heights), min(heights)
            if max_h == 0:
                return 0.0
            
            local_imbalance = (max_h - min_h) / max_h
            child_imbalances = [calculate_imbalance(child) for child in node.children]
            avg_child_imbalance = np.mean(child_imbalances) if child_imbalances else 0.0
            
            return 0.5 * local_imbalance + 0.5 * avg_child_imbalance
        
        return calculate_imbalance(self.ast_root)
    
    def _calculate_avg_path_length(self) -> float:
        """Calcola la lunghezza media del percorso dalla radice alle foglie."""
        path_lengths = []
        
        def traverse(node: ASTNode, depth: int):
            if not node.children:
                path_lengths.append(depth)
            else:
                for child in node.children:
                    traverse(child, depth + 1)
        
        traverse(self.ast_root, 0)
        return np.mean(path_lengths) if path_lengths else 0.0
    
    def _detect_patterns(self):
        """Rileva pattern comuni nell'AST."""
        operator_seq = []
        
        def extract_operators(node: ASTNode):
            if node.node_type in ['Add', 'Mul', 'Pow', 'Div']:
                operator_seq.append(node.node_type)
            for child in node.children:
                extract_operators(child)
        
        extract_operators(self.ast_root)
        self.advanced_features.operator_sequences = operator_seq
        
        if len(operator_seq) >= 3:
            trigrams = ['-'.join(operator_seq[i:i+3]) for i in range(len(operator_seq)-2)]
            self.advanced_features.common_subpatterns = dict(Counter(trigrams))
        
        def count_nested_ops(node: ASTNode, parent_op: Optional[str] = None):
            if node.node_type in ['Add', 'Mul', 'Pow']:
                if parent_op:
                    key = f"{parent_op}->{node.node_type}"
                    self.advanced_features.nested_operations[key] = \
                        self.advanced_features.nested_operations.get(key, 0) + 1
                
                for child in node.children:
                    count_nested_ops(child, node.node_type)
            else:
                for child in node.children:
                    count_nested_ops(child, parent_op)
        
        count_nested_ops(self.ast_root)
    
    def _detect_mathematical_types(self):
        """Rileva tipi di funzioni matematiche presenti."""
        def traverse(node: ASTNode):
            node_name = node.node_type.lower()
            
            if node_name in self.TRIG_FUNCS:
                self.advanced_features.has_trigonometric = True
            if node_name in self.LOG_FUNCS:
                self.advanced_features.has_logarithmic = True
            if node_name in ['exp'] or (node_name == 'pow' and node.children):
                self.advanced_features.has_exponential = True
            if node_name in self.TRANSCENDENTAL_FUNCS:
                self.advanced_features.has_transcendental = True
            
            for child in node.children:
                traverse(child)
        
        traverse(self.ast_root)
        
        try:
            if self.expr.is_polynomial():
                symbols = list(self.expr.free_symbols)
                if symbols:
                    self.advanced_features.polynomial_degree = sp.degree(self.expr, symbols[0])
        except:
            pass
    
    def _detect_symmetry(self):
        """Rileva simmetrie nell'AST."""
        subtree_hashes = defaultdict(int)
        
        def hash_subtree(node: ASTNode) -> str:
            if not node.children:
                return f"{node.node_type}:{node.value}"
            
            child_hashes = sorted([hash_subtree(child) for child in node.children])
            return f"{node.node_type}({','.join(child_hashes)})"
        
        def count_subtrees(node: ASTNode):
            subtree_hash = hash_subtree(node)
            subtree_hashes[subtree_hash] += 1
            
            for child in node.children:
                count_subtrees(child)
        
        count_subtrees(self.ast_root)
        
        repeated = sum(1 for count in subtree_hashes.values() if count > 1)
        self.advanced_features.has_repeated_subexpressions = repeated > 0
        
        if self.ast_root.node_type in ['Add', 'Mul'] and len(self.ast_root.children) == 2:
            left_hash = hash_subtree(self.ast_root.children[0])
            right_hash = hash_subtree(self.ast_root.children[1])
            self.advanced_features.is_symmetric = (left_hash == right_hash)
    
    def get_structural_features(self) -> Dict[str, Any]:
        """Calcola features strutturali dell'AST."""
        if not self.ast_root:
            return {}
        
        features = {}
        node_depths = []
        node_types = []
        node_children_counts = []
        all_nodes = []
        leaf_nodes = []
        
        def traverse(node: ASTNode):
            node_depths.append(node.depth)
            node_types.append(node.node_type)
            node_children_counts.append(len(node.children))
            all_nodes.append(node)
            
            if not node.children:
                leaf_nodes.append(node)
            
            for child in node.children:
                traverse(child)
        
        traverse(self.ast_root)
        
        features['max_depth'] = max(node_depths) if node_depths else 0
        features['avg_depth'] = float(np.mean(node_depths)) if node_depths else 0
        features['std_depth'] = float(np.std(node_depths)) if node_depths else 0
        features['total_nodes'] = len(all_nodes)
        features['leaf_nodes'] = len(leaf_nodes)
        features['internal_nodes'] = len(all_nodes) - len(leaf_nodes)
        
        non_leaf_nodes = [count for count in node_children_counts if count > 0]
        features['avg_branching_factor'] = float(np.mean(non_leaf_nodes)) if non_leaf_nodes else 0
        features['max_branching_factor'] = max(node_children_counts) if node_children_counts else 0
        features['branching_variance'] = float(np.var(non_leaf_nodes)) if non_leaf_nodes else 0
        
        type_counts = Counter(node_types)
        features['node_type_counts'] = dict(type_counts)
        features['node_type_diversity'] = len(type_counts)
        features['node_type_entropy'] = float(self._calculate_entropy(list(type_counts.values())))
        
        operation_nodes = [n for n in all_nodes if n.node_type in ['Add', 'Mul', 'Pow', 'Div', 'Sub']]
        features['operation_complexity'] = float(len(operation_nodes) / max(len(all_nodes), 1))
        features['operation_count'] = len(operation_nodes)
        
        depth_counts = Counter(node_depths)
        features['max_width'] = max(depth_counts.values()) if depth_counts else 0
        features['avg_width'] = float(np.mean(list(depth_counts.values()))) if depth_counts else 0
        
        return features
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calcola l'entropia di Shannon."""
        total = sum(counts)
        if total == 0:
            return 0.0
        
        probabilities = [count / total for count in counts]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def get_numerical_components(self) -> Dict[str, Any]:
        """Estrae e analizza componenti numerici con analisi avanzata."""
        if not self.ast_root:
            return {}
        
        components = {
            'coefficients': [],
            'exponents': [],
            'integers': [],
            'rationals': [],
            'constants': []
        }
        
        def extract_numerical(node: ASTNode):
            try:
                if node.node_type == 'Integer':
                    if isinstance(node.value, (int, float)):
                        value = int(node.value)
                        components['integers'].append(value)
                        components['coefficients'].append(float(value))
                elif node.node_type == 'Float':
                    if isinstance(node.value, (int, float)):
                        components['coefficients'].append(float(node.value))
                elif node.node_type == 'Rational':
                    if node.value is not None:
                        try:
                            float_val = float(node.value)
                            components['rationals'].append(str(node.value))
                            components['coefficients'].append(float_val)
                        except:
                            pass
                elif node.node_type in ['Pi', 'Exp1', 'One', 'Zero', 'Half', 'NegativeOne']:
                    components['constants'].append(node.node_type)
                elif node.node_type == 'Pow' and len(node.children) == 2:
                    exp_child = node.children[1]
                    if exp_child.value is not None and isinstance(exp_child.value, (int, float)):
                        components['exponents'].append(float(exp_child.value))
            except Exception:
                pass
            
            for child in node.children:
                extract_numerical(child)
        
        extract_numerical(self.ast_root)
        
        analysis = {}
        
        if components['coefficients']:
            coeffs = [c for c in components['coefficients'] if isinstance(c, (int, float))]
            if coeffs:
                analysis['coefficients'] = {
                    'count': len(coeffs),
                    'mean': float(np.mean(coeffs)),
                    'std': float(np.std(coeffs)),
                    'min': float(min(coeffs)),
                    'max': float(max(coeffs)),
                    'range': float(max(coeffs) - min(coeffs)),
                    'unique_count': len(set(coeffs))
                }
        
        if components['exponents']:
            exps = [e for e in components['exponents'] if isinstance(e, (int, float))]
            if exps:
                analysis['exponents'] = {
                    'count': len(exps),
                    'unique_count': len(set(exps)),
                    'max_degree': float(max(exps)),
                    'has_fractional': any(e % 1 != 0 for e in exps),
                    'has_negative': any(e < 0 for e in exps)
                }
        
        if components['integers']:
            ints = [i for i in components['integers'] if isinstance(i, int)]
            if ints:
                analysis['integers'] = {
                    'count': len(ints),
                    'has_zero': 0 in ints,
                    'has_one': 1 in ints,
                    'has_negative': any(i < 0 for i in ints)
                }
        
        if components['constants']:
            const_counts = Counter(components['constants'])
            analysis['mathematical_constants'] = dict(const_counts)
        
        return analysis
    
    def get_complete_analysis(self) -> Dict[str, Any]:
        """Restituisce tutte le analisi incluse quelle avanzate."""
        advanced_dict = {
            'tree_imbalance': float(self.advanced_features.tree_imbalance),
            'average_path_length': float(self.advanced_features.average_path_length),
            'has_transcendental': self.advanced_features.has_transcendental,
            'has_trigonometric': self.advanced_features.has_trigonometric,
            'has_logarithmic': self.advanced_features.has_logarithmic,
            'has_exponential': self.advanced_features.has_exponential,
            'polynomial_degree': self.advanced_features.polynomial_degree,
            'is_symmetric': self.advanced_features.is_symmetric,
            'has_repeated_subexpressions': self.advanced_features.has_repeated_subexpressions,
            'operator_sequences': self.advanced_features.operator_sequences,
            'common_subpatterns': self.advanced_features.common_subpatterns,
            'nested_operations': self.advanced_features.nested_operations
        }
        
        return {
            'raw_expression': self.raw_expr_str,
            'ast_root_type': self.ast_root.node_type if self.ast_root else None,
            'structural_features': self.structural_features,
            'numerical_components': self.numerical_components,
            'advanced_features': advanced_dict,
            'parsing_success': self.ast_root is not None
        }

# DIZIONARIO GLOBALE per caratteri greci e simboli speciali
GREEK_TO_LATIN = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lamda',  # Uso 'lamda' invece di 'lambda' per evitare conflitti
    'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
    'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
    'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
    'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta', 'Ε': 'Epsilon',
    'Ζ': 'Zeta', 'Η': 'Eta', 'Θ': 'Theta', 'Ι': 'Iota', 'Κ': 'Kappa',
    'Λ': 'Lambda', 'Μ': 'Mu', 'Ν': 'Nu', 'Ξ': 'Xi', 'Ο': 'Omicron',
    'Π': 'Pi', 'Ρ': 'Rho', 'Σ': 'Sigma', 'Τ': 'Tau', 'Υ': 'Upsilon',
    'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega',
    # Simboli fisici speciali
    'ℏ': 'hbar',  # h-bar (costante di Planck ridotta)
    'ℎ': 'h',  # Planck constant
    # Unicode subscripts
    '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4',
    '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
    '₊': '_plus', '₋': '_minus', 'ₓ': '_x', 'ᵢ': '_i', 'ⱼ': '_j',
    # Unicode superscripts 
    '²': '**2', '³': '**3', '⁴': '**4', '⁰': '**0', '¹': '**1'
}

def convert_greek_to_latin(text: str) -> str:
    """Converte TUTTI i caratteri greci in equivalenti latini."""
    result = text
    for greek, latin in GREEK_TO_LATIN.items():
        result = result.replace(greek, latin)
    return result

def extract_expression_from_equation(equation_str: str) -> str:
    """Estrae l'espressione da un'equazione completa."""
    equation_str = equation_str.strip()
    
    # Se contiene '=', gestisci in modo intelligente
    if '=' in equation_str:
        parts = equation_str.split('=', 1)
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            
            # Se la parte destra è vuota o banale, usa la sinistra
            if not right or right in ['0', '1']:
                return left if left else equation_str
            
            # Se la parte sinistra è una singola variabile, usa la destra
            if re.match(r'^[a-zA-Z_]\w*$', left):
                return right
            
            # Altrimenti, ritorna la parte destra
            return right
    
    return equation_str

def ultimate_robust_sympy_parse(equation_str: str, debug: bool = False) -> Tuple[Optional[Any], Dict[str, str]]:
    """
    Parsing DEFINITIVO con debug completo.
    Ritorna tupla (risultato, debug_info)
    """
    debug_info = {
        'original': equation_str,
        'after_greek': '',
        'after_extract': '',
        'after_preprocess': '',
        'final_cleaned': '',
        'error': ''
    }
    
    try:
        # STEP 1: Converti IMMEDIATAMENTE i caratteri greci e speciali
        equation_str = convert_greek_to_latin(equation_str)
        debug_info['after_greek'] = equation_str
        
        # STEP 2: Gestisci casi speciali PRIMA dell'estrazione
        # Rimuovi ellipsis che causa problemi
        equation_str = equation_str.replace('...', '')
        equation_str = equation_str.replace('…', '')  # Unicode ellipsis
        
        # Gestisci B.E. come caso speciale
        equation_str = re.sub(r'B\.E\.', 'B_E', equation_str)
        
        # STEP 3: Estrai espressione se necessario
        expression_str = extract_expression_from_equation(equation_str)
        debug_info['after_extract'] = expression_str
        
        # STEP 4: Preprocessing speciale
        # Gestisci nabla squared (laplaciano)
        expression_str = expression_str.replace('∇²', 'nabla2')
        expression_str = expression_str.replace('∇', 'nabla')
        
        # Gestisci derivate parziali quadrate
        expression_str = re.sub(r'∂²/∂(\w+)²', r'd2_d\1_2', expression_str)
        expression_str = re.sub(r'\(∂/∂(\w+)\)²', r'(d_d\1)**2', expression_str)
        expression_str = re.sub(r'∂²', 'd2', expression_str)
        
        # Gestisci integrali
        expression_str = re.sub(r'∮\s*\((.*?)\)', r'Integral(\1, x)', expression_str)  # Integrali di linea
        expression_str = re.sub(r'∫\s*\((.*?)\)', r'Integral(\1, x)', expression_str)
        expression_str = re.sub(r'∫\s*([^+\-*/\s]+)', r'Integral(\1, x)', expression_str)
        expression_str = expression_str.replace('∮', 'Integral')  # Integrale di linea
        expression_str = expression_str.replace('∫', 'Integral')
        
        # Altri simboli
        expression_str = expression_str.replace('∂', 'd')
        expression_str = expression_str.replace('×', '*')
        expression_str = expression_str.replace('·', '*')
        expression_str = expression_str.replace('⃗', '_vec')
        expression_str = expression_str.replace('√', 'sqrt')
        expression_str = expression_str.replace(';', ',')
        expression_str = expression_str.replace("'", "_prime")
        
        # Gestisci valori assoluti
        parts = expression_str.split('|')
        if len(parts) > 1:
            new_expr = parts[0]
            for i in range(1, len(parts)):
                if i % 2 == 1:
                    new_expr += 'Abs(' + parts[i]
                else:
                    new_expr += ')' + parts[i]
            expression_str = new_expr
        
        debug_info['after_preprocess'] = expression_str
        
        # STEP 5: Cleaning finale
        # Rimuovi caratteri problematici MA mantieni underscore
        cleaned = re.sub(r'[^\w\s\+\-\*\/\^\(\)\[\]\{\}\.,=<>!~_]', '', expression_str)
        
        # Sostituzioni standard
        cleaned = cleaned.replace('^', '**')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix moltiplicazioni implicite
        cleaned = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', cleaned)
        cleaned = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', cleaned)
        cleaned = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1*\2', cleaned)
        cleaned = re.sub(r'\)\s*\(', r')*(', cleaned)
        cleaned = re.sub(r'(\d)\s*\(', r'\1*(', cleaned)
        cleaned = re.sub(r'\)\s*([a-zA-Z])', r')*\1', cleaned)
        
        # Fix frazioni
        cleaned = re.sub(r'(\d+)/(\d+)', r'(\1/\2)', cleaned)
        
        cleaned = cleaned.strip()
        debug_info['final_cleaned'] = cleaned
        
        if not cleaned:
            debug_info['error'] = 'Empty after cleaning'
            return None, debug_info
        
        # STEP 6: Parsing con dizionario completo
        # Crea dizionario con TUTTI i simboli possibili
        symbols_dict = {
            # Funzioni
            'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'pi': sp.pi, 'e': sp.E,
            'Integral': sp.Integral, 'Abs': sp.Abs,
            # Costanti
            'oo': sp.oo, 'zoo': sp.zoo,
            # Simboli speciali per equazioni problematiche
            'hbar': sp.Symbol('hbar'),
            'B_E': sp.Symbol('B_E'),
            'nabla': sp.Symbol('nabla'),
            'nabla2': sp.Symbol('nabla2'),
            'd2_dt_2': sp.Symbol('d2_dt_2'),
            'd_dt': sp.Symbol('d_dt'),
            'flux': sp.Symbol('flux'),
            'density': sp.Symbol('density'),
            'M': sp.Symbol('M'),
            'R_total': sp.Symbol('R_total')
        }
        
        # Aggiungi TUTTI i simboli greci convertiti
        for greek_latin in set(GREEK_TO_LATIN.values()):
            if not greek_latin.startswith('**') and not greek_latin.startswith('_'):
                symbols_dict[greek_latin] = sp.Symbol(greek_latin)
        
        # Aggiungi lettere singole e con underscore
        for letter in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            symbols_dict[letter] = sp.Symbol(letter)
            # Con numeri
            for num in '0123456789':
                symbols_dict[f'{letter}_{num}'] = sp.Symbol(f'{letter}_{num}')
                symbols_dict[f'{letter}{num}'] = sp.Symbol(f'{letter}_{num}')
            # Altri suffissi
            for suffix in ['_x', '_y', '_z', '_i', '_j', '_f', '_prime', '_vec', '_total']:
                symbols_dict[f'{letter}{suffix}'] = sp.Symbol(f'{letter}{suffix}')
        
        # Simboli speciali
        special_symbols = ['dl', 'dQ', 'dt', 'dx', 'dy', 'dz', 'dr', 'dtheta', 'dphi']
        for sym in special_symbols:
            symbols_dict[sym] = sp.Symbol(sym)
        
        # Prova parsing
        try:
            # Prima con parse_expr
            transformations = (
                sp.parsing.sympy_parser.standard_transformations +
                (sp.parsing.sympy_parser.implicit_multiplication_application,)
            )
            result = sp.parse_expr(cleaned, local_dict=symbols_dict, transformations=transformations)
            return result, debug_info
        except:
            # Fallback a sympify
            try:
                result = sp.sympify(cleaned, locals=symbols_dict)
                return result, debug_info
            except Exception as e:
                debug_info['error'] = str(e)
                return None, debug_info
                
    except Exception as e:
        debug_info['error'] = f'Unexpected error: {str(e)}'
        return None, debug_info

def make_json_serializable(obj):
    """Converte oggetti non serializzabili in formati JSON-compatibili."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    elif obj is None:
        return None
    else:
        return str(obj)

def analyze_by_branch(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analizza le equazioni raggruppate per branch."""
    branch_stats = defaultdict(lambda: {
        'count': 0,
        'avg_depth': [],
        'avg_nodes': [],
        'avg_complexity': [],
        'common_patterns': Counter(),
        'math_types': Counter()
    })
    
    for analysis in analyses:
        branch = analysis.get('branch', 'Unknown')
        stats = branch_stats[branch]
        stats['count'] += 1
        
        struct_feat = analysis.get('structural_features', {})
        if struct_feat:
            stats['avg_depth'].append(struct_feat.get('max_depth', 0))
            stats['avg_nodes'].append(struct_feat.get('total_nodes', 0))
            stats['avg_complexity'].append(struct_feat.get('operation_complexity', 0))
        
        adv_feat = analysis.get('advanced_features', {})
        if adv_feat:
            if adv_feat.get('has_trigonometric'):
                stats['math_types']['trigonometric'] += 1
            if adv_feat.get('has_exponential'):
                stats['math_types']['exponential'] += 1
            if adv_feat.get('has_logarithmic'):
                stats['math_types']['logarithmic'] += 1
            
            for pattern in adv_feat.get('common_subpatterns', {}).keys():
                stats['common_patterns'][pattern] += 1
    
    result = {}
    for branch, stats in branch_stats.items():
        result[branch] = {
            'count': stats['count'],
            'avg_max_depth': float(np.mean(stats['avg_depth'])) if stats['avg_depth'] else 0,
            'avg_total_nodes': float(np.mean(stats['avg_nodes'])) if stats['avg_nodes'] else 0,
            'avg_operation_complexity': float(np.mean(stats['avg_complexity'])) if stats['avg_complexity'] else 0,
            'most_common_patterns': stats['common_patterns'].most_common(5),
            'mathematical_types': dict(stats['math_types'])
        }
    
    return result

def analyze_parsing_failures(failed_parses: List[Dict]) -> Dict[str, Any]:
    """Analizza i fallimenti di parsing per identificare pattern comuni."""
    if not failed_parses:
        return {}
    
    error_patterns = Counter()
    problematic_chars = Counter()
    equation_lengths = []
    contains_equals = 0
    
    for fail in failed_parses:
        error_msg = fail.get('error', 'Unknown')
        if 'Empty' in error_msg:
            error_patterns['empty_equation'] += 1
        elif 'SyntaxError' in error_msg:
            error_patterns['syntax_error'] += 1
        elif 'NameError' in error_msg:
            error_patterns['undefined_variable'] += 1
        elif 'parsing strategies failed' in error_msg:
            error_patterns['all_strategies_failed'] += 1
        else:
            error_patterns['other'] += 1
        
        equation = fail.get('equation', '')
        
        if '=' in equation:
            contains_equals += 1
        
        for char in equation:
            if not char.isalnum() and char not in '+-*/^()[]{}., ':
                problematic_chars[char] += 1
        
        equation_lengths.append(len(equation))
    
    return {
        'error_distribution': dict(error_patterns),
        'problematic_characters': dict(problematic_chars.most_common(10)),
        'average_equation_length': np.mean(equation_lengths) if equation_lengths else 0,
        'equations_with_equals': contains_equals
    }

def run_structural_analysis(equations_data: List[Dict], save_results: bool = True) -> Dict[str, Any]:
    """Funzione principale per l'analisi strutturale completa con parsing migliorato."""
    print("🚀 STARTING COMPREHENSIVE STRUCTURAL ANALYSIS...")
    print(f"📊 Total equations to analyze: {len(equations_data)}")
    
    successful_parses = []
    failed_parses = []
    
    total = len(equations_data)
    checkpoint = max(1, total // 10)
    
    for i, law_data in enumerate(equations_data):
        equation_str = law_data.get('equation', '')
        law_id = law_data.get('id', f'law_{i}')
        
        if i > 0 and i % checkpoint == 0:
            progress = (i / total) * 100
            success_rate = (len(successful_parses) / (i+1)) * 100 if i > 0 else 0
            print(f"\n📈 Progress: {progress:.1f}% ({i}/{total} equations processed)")
            print(f"   ✅ Successful: {len(successful_parses)} ({success_rate:.1f}%)")
            print(f"   ❌ Failed: {len(failed_parses)}")
        
        if not equation_str:
            failed_parses.append({
                'law_id': law_id,
                'equation': equation_str,
                'error': 'Empty equation string'
            })
            continue
        
        try:
            # Usa il parser definitivo con debug
            sympy_expr, debug_info = ultimate_robust_sympy_parse(equation_str, debug=True)
            
            if sympy_expr is None:
                failed_parses.append({
                    'law_id': law_id,
                    'equation': equation_str,
                    'error': debug_info.get('error', 'All parsing strategies failed'),
                    'debug_info': debug_info
                })
                continue
            
            parser = EquationParser(sympy_expr)
            analysis = parser.get_complete_analysis()
            
            if analysis['parsing_success']:
                analysis['law_id'] = law_id
                analysis['branch'] = law_data.get('branch', 'Unknown')
                analysis['category'] = law_data.get('category', 'Unknown')
                analysis['difficulty'] = law_data.get('difficulty', 'Unknown')
                analysis['original_equation'] = equation_str
                successful_parses.append(analysis)
            else:
                failed_parses.append({
                    'law_id': law_id,
                    'equation': equation_str,
                    'error': 'AST parsing failed',
                    'debug_info': debug_info
                })
                
        except Exception as e:
            failed_parses.append({
                'law_id': law_id,
                'equation': equation_str,
                'error': f'Error: {str(e)}'
            })
    
    success_rate = (len(successful_parses) / total * 100) if total > 0 else 0
    
    print(f"\n✅ Analysis completed!")
    print(f"   📊 Total equations: {total}")
    print(f"   ✅ Successful parses: {len(successful_parses)} ({success_rate:.1f}%)")
    print(f"   ❌ Failed parses: {len(failed_parses)} ({100-success_rate:.1f}%)")
    
    branch_analysis = analyze_by_branch(successful_parses)
    
    if successful_parses:
        all_depths = [a['structural_features']['max_depth'] for a in successful_parses]
        all_nodes = [a['structural_features']['total_nodes'] for a in successful_parses]
        all_complexities = [a['structural_features']['operation_complexity'] for a in successful_parses]
        
        global_stats = {
            'depth': {
                'mean': float(np.mean(all_depths)),
                'std': float(np.std(all_depths)),
                'min': int(min(all_depths)),
                'max': int(max(all_depths)),
                'median': float(np.median(all_depths))
            },
            'nodes': {
                'mean': float(np.mean(all_nodes)),
                'std': float(np.std(all_nodes)),
                'min': int(min(all_nodes)),
                'max': int(max(all_nodes)),
                'median': float(np.median(all_nodes))
            },
            'complexity': {
                'mean': float(np.mean(all_complexities)),
                'std': float(np.std(all_complexities)),
                'min': float(min(all_complexities)),
                'max': float(max(all_complexities)),
                'median': float(np.median(all_complexities))
            }
        }
    else:
        global_stats = {}
    
    results = {
        'successful_analyses': successful_parses,
        'failed_analyses': failed_parses,
        'summary_stats': {
            'total_equations': len(equations_data),
            'successful_count': len(successful_parses),
            'failed_count': len(failed_parses),
            'success_rate': success_rate / 100
        },
        'branch_analysis': branch_analysis,
        'global_statistics': global_stats
    }
    
    if save_results and successful_parses:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_file = f'ast_analysis_results_{timestamp}.json'
        json_safe_results = make_json_serializable(results)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not save full JSON file: {e}")
            
            try:
                simplified_results = {
                    'summary_stats': make_json_serializable(results['summary_stats']),
                    'branch_analysis': make_json_serializable(results['branch_analysis']),
                    'global_statistics': make_json_serializable(results['global_statistics']),
                    'failed_count': len(results['failed_analyses'])
                }
                simple_file = f'ast_analysis_summary_{timestamp}.json'
                with open(simple_file, 'w', encoding='utf-8') as f:
                    json.dump(simplified_results, f, indent=2, ensure_ascii=False)
                print(f"💾 Simplified summary saved to: {simple_file}")
            except:
                print("❌ Could not save results to JSON")
        
        create_visualizations(successful_parses)
        
        try:
            report_file = f'ast_analysis_report_{timestamp}.txt'
            generate_text_report(results, report_file)
            print(f"📄 Text report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save text report: {e}")
    
    return results

def create_visualizations(analyses: List[Dict[str, Any]], output_dir: str = 'ast_analysis_plots'):
    """Crea visualizzazioni dei risultati dell'analisi."""
    if not PLOTTING_AVAILABLE:
        print("⚠️  Skipping visualizations (matplotlib not available)")
        return
    
    print("📊 Creating visualizations...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepara dati
    depths = []
    nodes = []
    complexities = []
    branches = []
    
    for analysis in analyses:
        struct_feat = analysis.get('structural_features', {})
        if struct_feat:
            depths.append(struct_feat.get('max_depth', 0))
            nodes.append(struct_feat.get('total_nodes', 0))
            complexities.append(struct_feat.get('operation_complexity', 0))
            branches.append(analysis.get('branch', 'Unknown'))
    
    # 1. Distribuzione profondità
    plt.figure(figsize=(10, 6))
    plt.hist(depths, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Max Depth')
    plt.ylabel('Frequency')
    plt.title('Distribution of AST Maximum Depth')
    plt.savefig(f'{output_dir}/depth_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot nodes vs depth
    plt.figure(figsize=(10, 6))
    plt.scatter(depths, nodes, alpha=0.5)
    plt.xlabel('Max Depth')
    plt.ylabel('Total Nodes')
    plt.title('AST Nodes vs Depth')
    plt.savefig(f'{output_dir}/nodes_vs_depth.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Complessità per branch
    branch_df = pd.DataFrame({
        'branch': branches,
        'complexity': complexities
    })
    
    plt.figure(figsize=(12, 6))
    branch_df.boxplot(column='complexity', by='branch', rot=45)
    plt.xlabel('Physics Branch')
    plt.ylabel('Operation Complexity')
    plt.title('Operation Complexity by Physics Branch')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_by_branch.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Visualizations saved to: ast_analysis_plots/")

def generate_text_report(results: Dict[str, Any], output_file: str):
    """Genera un report testuale dettagliato."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AST ANALYSIS REPORT FOR PHYSICS EQUATIONS DATABASE\n")
        f.write("=" * 80 + "\n\n")
        
        summary = results['summary_stats']
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Equations: {summary['total_equations']}\n")
        f.write(f"Successfully Parsed: {summary['successful_count']} ({summary['success_rate']*100:.1f}%)\n")
        f.write(f"Failed Parsing: {summary['failed_count']}\n\n")
        
        if results['global_statistics']:
            f.write("GLOBAL STATISTICS\n")
            f.write("-" * 40 + "\n")
            for metric, stats in results['global_statistics'].items():
                f.write(f"\n{metric.upper()}:\n")
                for stat, value in stats.items():
                    f.write(f"  {stat}: {value:.2f}\n")
        
        f.write("\n\nBRANCH ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for branch, stats in sorted(results['branch_analysis'].items()):
            f.write(f"\n{branch}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Avg Depth: {stats['avg_max_depth']:.2f}\n")
            f.write(f"  Avg Nodes: {stats['avg_total_nodes']:.2f}\n")
            f.write(f"  Avg Complexity: {stats['avg_operation_complexity']:.3f}\n")
            
            if stats['most_common_patterns']:
                f.write("  Most Common Patterns:\n")
                for pattern, count in stats['most_common_patterns']:
                    f.write(f"    {pattern}: {count}\n")
            
            if stats['mathematical_types']:
                f.write("  Mathematical Types:\n")
                for mtype, count in stats['mathematical_types'].items():
                    f.write(f"    {mtype}: {count}\n")
        
        if results['failed_analyses']:
            f.write("\n\nFAILED PARSING DETAILS\n")
            f.write("-" * 40 + "\n")
            for i, fail in enumerate(results['failed_analyses'][:20]):
                f.write(f"\n{i+1}. ID: {fail['law_id']}\n")
                f.write(f"   Equation: {fail['equation']}\n")
                f.write(f"   Error: {fail['error']}\n")
                
                if 'debug_info' in fail:
                    debug = fail['debug_info']
                    f.write(f"   Debug Info:\n")
                    f.write(f"     After Greek: {debug.get('after_greek', 'N/A')}\n")
                    f.write(f"     After Extract: {debug.get('after_extract', 'N/A')}\n")
                    f.write(f"     Final Cleaned: {debug.get('final_cleaned', 'N/A')}\n")

# Main execution
if __name__ == "__main__":
    print("🔬 AST ANALYZER - FINAL 100% VERSION WITH DEBUG")
    print("=" * 80)
    
    try:
        with open('final_physics_database.json', 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        laws = database.get('laws', [])
        print(f"✅ Loaded {len(laws)} equations from final_physics_database.json")
        
        # TEST: Verifica parsing delle equazioni problematiche
        print("\n🧪 TESTING problematic equations...")
        test_equations = [
            "γ * m * c^2",
            "f_0 * sqrt((1 + β)/(1 - β))",
            "sqrt(γ * R * T / M)",
            "N_0 * exp(-λ * t)",
            "I * ∫(dl × B)",
            "∫(dQ / T)"
        ]
        
        for eq in test_equations:
            result, debug = ultimate_robust_sympy_parse(eq, debug=True)
            print(f"\n📝 Testing: {eq}")
            print(f"   After Greek: {debug['after_greek']}")
            print(f"   Final cleaned: {debug['final_cleaned']}")
            print(f"   Result: {'✅ SUCCESS' if result else '❌ FAILED'}")
            if result:
                print(f"   Parsed as: {result}")
            else:
                print(f"   Error: {debug['error']}")
        
        print("\n" + "="*80)
        print("🔬 Running COMPLETE analysis on ALL equations...")
        results = run_structural_analysis(laws, save_results=True)
        
        print("\n" + "=" * 80)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 80)
        
        if results['successful_analyses']:
            most_complex = max(results['successful_analyses'], 
                             key=lambda x: x['structural_features']['total_nodes'])
            print(f"\n🏆 Most Complex Equation:")
            print(f"   ID: {most_complex['law_id']}")
            print(f"   Equation: {most_complex['original_equation']}")
            print(f"   Total Nodes: {most_complex['structural_features']['total_nodes']}")
            print(f"   Max Depth: {most_complex['structural_features']['max_depth']}")
            
            deepest = max(results['successful_analyses'], 
                         key=lambda x: x['structural_features']['max_depth'])
            print(f"\n🏔️ Deepest Equation:")
            print(f"   ID: {deepest['law_id']}")
            print(f"   Equation: {deepest['original_equation']}")
            print(f"   Max Depth: {deepest['structural_features']['max_depth']}")
        
        if results['failed_analyses']:
            print(f"\n❌ FAILURE ANALYSIS:")
            failure_analysis = analyze_parsing_failures(results['failed_analyses'])
            
            print(f"\n📊 Error Distribution:")
            for error_type, count in failure_analysis['error_distribution'].items():
                print(f"   {error_type}: {count}")
            
            print(f"\n⚠️  Equations with '=' sign: {failure_analysis['equations_with_equals']}")
            
            if failure_analysis['problematic_characters']:
                print(f"\n⚠️  Problematic Characters:")
                for char, count in failure_analysis['problematic_characters'].items():
                    print(f"   '{char}' (Unicode: {ord(char)}): {count} occurrences")
            
            print(f"\n📐 Failed equations average length: {failure_analysis['average_equation_length']:.1f}")
            
            print(f"\n🔍 Detailed Failed Equations (showing {min(5, len(results['failed_analyses']))} of {len(results['failed_analyses'])}):")
            for i, fail in enumerate(results['failed_analyses'][:5]):
                print(f"\n{i+1}. {fail['law_id']}: {fail['equation']}")
                if 'debug_info' in fail:
                    debug = fail['debug_info']
                    print(f"   After Greek conversion: {debug.get('after_greek', 'N/A')}")
                    print(f"   After preprocessing: {debug.get('after_preprocess', 'N/A')}")
                    print(f"   Final cleaned: {debug.get('final_cleaned', 'N/A')}")
                    print(f"   Error: {debug.get('error', 'Unknown')}")
                    
    except FileNotFoundError:
        print("❌ ERROR: final_physics_database.json not found!")
        print("Make sure the file is in the same directory as this script.")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()