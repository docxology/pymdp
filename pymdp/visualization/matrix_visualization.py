import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from matplotlib.gridspec import GridSpec
import networkx as nx
import os

logger = logging.getLogger(__name__)

def save_all_visualizations(matrices, save_dir, model_name="model"):
    """
    Generate and save all visualizations for a generative model
    
    Parameters
    ----------
    matrices : dict
        Dictionary containing model matrices (A, B, D, etc.)
    save_dir : str
        Directory to save visualizations
    model_name : str, optional
        Name prefix for saved files
    """
    logger.info(f"Generating all visualizations for {model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot A matrices
    if 'A' in matrices:
        A_dir = os.path.join(save_dir, "A_matrices")
        os.makedirs(A_dir, exist_ok=True)
        plot_A_matrices(matrices['A'], save_dir=A_dir)
        
    # Plot B matrices
    if 'B' in matrices:
        B_dir = os.path.join(save_dir, "B_matrices")
        os.makedirs(B_dir, exist_ok=True)
        plot_B_matrices(matrices['B'], save_dir=B_dir)
        
    # Plot model structure
    if 'A' in matrices:
        plot_model_graph(matrices['A'], 
                        save_to=os.path.join(save_dir, "model_structure.png"))
        
    logger.info(f"All visualizations saved to {save_dir}")

def plot_matrix_heatmap(matrix, title="", xlabel="", ylabel="", cmap='viridis', 
                       annot=True, fmt='.2f', figsize=(8, 6), save_to=None):
    """Plot matrix as heatmap"""
    logger.debug(f"Plotting matrix heatmap: {title}")
    logger.debug(f"Matrix shape: {matrix.shape}")
    
    if len(matrix.shape) == 3:
        # For 3D matrices, create subplots for each slice
        n_slices = matrix.shape[2]
        fig, axes = plt.subplots(1, n_slices, figsize=(figsize[0]*n_slices, figsize[1]))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
            
        for i, ax in enumerate(axes):
            sns.heatmap(matrix[:,:,i], cmap=cmap, annot=annot, fmt=fmt, 
                       square=True, cbar=True, ax=ax)
            ax.set_title(f"{title} (Slice {i})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if i == 0 else '')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
    else:
        plt.figure(figsize=figsize)
        ax = sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=fmt, 
                        square=True, cbar_kws={'label': 'Value'})
        plt.title(title, pad=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_to:
        logger.info(f"Saving heatmap to {save_to}")
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.close()

def plot_A_matrices(A, modality_names=None, state_names=None, save_dir=None):
    """Plot observation model matrices"""
    logger.info("Plotting A matrices")
    
    if modality_names is None:
        modality_names = [f"Modality {i}" for i in range(len(A))]
    if state_names is None:
        state_names = [f"State {i}" for i in range(A[0].shape[1])]
    
    for i, A_i in enumerate(A):
        logger.debug(f"Plotting A matrix for {modality_names[i]}")
        logger.debug(f"Matrix shape: {A_i.shape}")
        
        save_path = f"{save_dir}/A_matrix_{modality_names[i]}.png" if save_dir else None
        
        plot_matrix_heatmap(
            A_i,
            title=f"Observation Model - {modality_names[i]}",
            xlabel="Hidden States\n" + "\n".join(state_names),
            ylabel="Observations",
            save_to=save_path
        )
        
        with np.printoptions(precision=3, suppress=True):
            logger.debug(f"Matrix values:\n{A_i}")

def plot_B_matrices(B, factor_names=None, action_names=None, save_dir=None):
    """Plot transition model matrices"""
    logger.info("Plotting B matrices")
    
    if factor_names is None:
        factor_names = [f"Factor {i}" for i in range(len(B))]
    
    for f, B_f in enumerate(B):
        logger.debug(f"Plotting B matrix for {factor_names[f]}")
        logger.debug(f"Matrix shape: {B_f.shape}")
        
        if len(B_f.shape) == 3:  # Controllable factor
            num_actions = B_f.shape[2]
            action_labels = action_names[f] if action_names else [f"Action {a}" for a in range(num_actions)]
            
            for a in range(num_actions):
                save_path = f"{save_dir}/B_matrix_{factor_names[f]}_action_{a}.png" if save_dir else None
                
                plot_matrix_heatmap(
                    B_f[:,:,a],
                    title=f"Transition Model - {factor_names[f]} ({action_labels[a]})",
                    xlabel="Current State",
                    ylabel="Next State",
                    save_to=save_path
                )
                
                with np.printoptions(precision=3, suppress=True):
                    logger.debug(f"Matrix values for action {a}:\n{B_f[:,:,a]}")
        else:
            save_path = f"{save_dir}/B_matrix_{factor_names[f]}.png" if save_dir else None
            
            plot_matrix_heatmap(
                B_f,
                title=f"Transition Model - {factor_names[f]}",
                xlabel="Current State",
                ylabel="Next State",
                save_to=save_path
            )
            
            with np.printoptions(precision=3, suppress=True):
                logger.debug(f"Matrix values:\n{B_f}")

def plot_belief_history(belief_hist, factor_names=None, save_to=None):
    """Plot belief evolution over time"""
    logger.info("Plotting belief history")
    
    if factor_names is None:
        factor_names = [f"Factor {i}" for i in range(len(belief_hist[0]))]
    
    num_factors = len(belief_hist[0])
    fig = plt.figure(figsize=(15, 3*num_factors))
    gs = GridSpec(num_factors, 1)
    
    for f in range(num_factors):
        logger.debug(f"Plotting beliefs for {factor_names[f]}")
        
        ax = fig.add_subplot(gs[f])
        belief_matrix = np.array([beliefs[f] for beliefs in belief_hist])
        
        im = ax.imshow(belief_matrix.T, aspect='auto', cmap='viridis')
        ax.set_title(f"Belief Evolution - {factor_names[f]}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("State")
        
        plt.colorbar(im, ax=ax, label="Probability")
        
        logger.debug(f"Final beliefs for {factor_names[f]}:")
        with np.printoptions(precision=3, suppress=True):
            logger.debug(f"{belief_matrix[-1]}")
    
    plt.tight_layout()
    
    if save_to:
        logger.info(f"Saving belief history plot to {save_to}")
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.close()

def plot_model_graph(A, factor_names=None, modality_names=None, save_to=None):
    """Plot model structure graph"""
    logger.info("Plotting model structure graph")
    
    try:
        if factor_names is None:
            factor_names = [f"Factor {i}" for i in range(len(A[0].shape[1:]))]
        if modality_names is None:
            modality_names = [f"Modality {i}" for i in range(len(A))]
            
        G = nx.DiGraph()
        
        logger.debug("Adding nodes to graph")
        G.add_nodes_from(factor_names, node_type='factor')
        G.add_nodes_from(modality_names, node_type='modality')
        
        logger.debug("Adding edges based on A matrix dependencies")
        for m, A_m in enumerate(A):
            factor_dims = len(A_m.shape) - 1
            for f in range(factor_dims):
                if A_m.shape[f+1] > 1:
                    G.add_edge(factor_names[f], modality_names[m], edge_type='observation')
                    logger.debug(f"Added edge: {factor_names[f]} -> {modality_names[m]}")
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        logger.debug("Drawing factor nodes")
        factor_nodes = [n for n,d in G.nodes(data=True) if d['node_type']=='factor']
        nx.draw_networkx_nodes(G, pos, nodelist=factor_nodes,
                             node_color='lightblue', node_size=2000, label='Factors')
        
        logger.debug("Drawing modality nodes")
        modality_nodes = [n for n,d in G.nodes(data=True) if d['node_type']=='modality']
        nx.draw_networkx_nodes(G, pos, nodelist=modality_nodes,
                             node_color='lightgreen', node_size=1500, label='Modalities')
        
        logger.debug("Drawing edges")
        obs_edges = [(u,v) for (u,v,d) in G.edges(data=True) if d['edge_type']=='observation']
        nx.draw_networkx_edges(G, pos, edgelist=obs_edges, edge_color='b', 
                             arrows=True, arrowsize=20)
        
        logger.debug("Adding node labels")
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title("Model Structure")
        plt.legend()
        plt.tight_layout()
        
        if save_to:
            logger.info(f"Saving model graph to {save_to}")
            plt.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.close()
            
        logger.info("Model graph plotting complete")
        
    except Exception as e:
        logger.error(f"Error plotting model graph: {str(e)}")
        raise
