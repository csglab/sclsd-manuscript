# various import statements
import numpy as np
import torch
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(42)

def random_walk(connectivity_matrix, start_cell, n_steps):
    """
    Perform a random walk on the graph defined by a sparse connectivity matrix.
    
    Parameters:
    connectivity_matrix (scipy.sparse.csr_matrix): The sparse connectivity matrix from scanpy
    start_cell (int): The index of the starting cell
    n_steps (int): Number of steps in the random walk
    
    Returns:
    np.array: Array of cell indices visited during the walk
    """
    n_cells = connectivity_matrix.shape[0]
    
    # Normalize the sparse connectivity matrix to create transition probabilities
    row_sums = np.array(connectivity_matrix.sum(axis=1)).flatten()
    transition_probs = sp.diags(1.0 / row_sums) @ connectivity_matrix
    
    # Initialize the walk
    walk = np.zeros(n_steps + 1, dtype=int)
    walk[0] = start_cell
    
    # Perform the walk
    for i in range(1, n_steps + 1):
        current_cell = walk[i-1]
        next_cell = np.random.choice(n_cells, p=transition_probs[current_cell].toarray().flatten())
        walk[i] = next_cell
    
    return walk

def plot_random_walks(adata, walks,rep, n_neighbors=64):

    
    # Get UMAP coordinates
    coords = adata.obsm[rep]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all cells
    if rep == 'X_tsne':
        sc.pl.tsne(adata, ax=ax, show=False, title='Random Walks on TSNE')
    if rep == 'X_umap':
        sc.pl.umap(adata, ax=ax, show=False, title='Random Walks on UMAP')
    
    # Define a color palette for the walks
    colors = sns.color_palette('husl', len(walks))
    
    labeled_start_end = False
    
    for idx, walk in enumerate(walks):
        # Extract coordinates for the current walk
        walk_coords = coords[walk]
        
        # Plot the walk
        ax.plot(walk_coords[:, 0], walk_coords[:, 1], '-', color=colors[idx], linewidth=0.8, alpha=0.6)
        ax.plot(walk_coords[:, 0], walk_coords[:, 1], '.', color=colors[idx], markersize=3, alpha=0.6)
        
        # Highlight start and end points, label them only once
        if not labeled_start_end:
            ax.plot(walk_coords[0, 0], walk_coords[0, 1], 'o', color='green', markersize=10, label='Start')
            ax.plot(walk_coords[-1, 0], walk_coords[-1, 1], 'o', color='magenta', markersize=10, label='End')
            labeled_start_end = True
        else:
            ax.plot(walk_coords[0, 0], walk_coords[0, 1], 'o', color='green', markersize=10)
            ax.plot(walk_coords[-1, 0], walk_coords[-1, 1], 'o', color='magenta', markersize=10)
    
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def generate_trajectory(connectivity, n_steps=128):
    start_cell = np.random.randint(connectivity.shape[0])
        
    return random_walk(connectivity, start_cell, n_steps -1)


def Prepare_simple_walks(n_steps, n_trajectories, connectivity):
    walks = []
    for i in range(n_trajectories):
        walk = generate_trajectory(connectivity = connectivity,n_steps = n_steps)
        walks.append(walk)
    return torch.from_numpy(np.array(walks))

def prepare_transition_matrix_gpu(connectivity):
    """
    Convert the sparse connectivity matrix into a dense transition probability
    matrix on GPU. Rows are normalized so that each sums to 1.
    """
    # Convert to a dense numpy array
    connectivity_dense = connectivity.toarray()
    # Normalize each row to sum to 1
    row_sums = connectivity_dense.sum(axis=1, keepdims=True)
    T = connectivity_dense / row_sums
    # Convert to a torch tensor on GPU
    T_tensor = torch.tensor(T, dtype=torch.float32, device="cuda")
    return T_tensor

def visualize_random_walks_on_umap(
    dyn_adata, paths, target_clusters, 
    cluster_key='clusters', n_walks=10, 
    figsize=(12, 8), alpha_walk=0.7, 
    linewidth=1.5, seed=42, rep="X_umap", filename=None,
    # --- new ---
    rasterize=True, rasterization_zorder=1, raster_dpi=300
):
    """
    Visualize random walks on UMAP starting from cells in specific clusters.
    Set `rasterize=True` to embed heavy artists as bitmaps inside the SVG/PDF.
    """

    # --- Validation ---
    if rep not in dyn_adata.obsm:
        raise KeyError(f"Embedding '{rep}' not found in dyn_adata.obsm")
    if cluster_key not in dyn_adata.obs:
        raise KeyError(f"Cluster key '{cluster_key}' not found in dyn_adata.obs")

    import numpy as np
    import matplotlib.pyplot as plt

    # Reproducibility
    np.random.seed(seed)

    # Data
    coords = dyn_adata.obsm[rep]
    if not isinstance(target_clusters, list):
        target_clusters = [target_clusters]
    clusters = dyn_adata.obs[cluster_key].astype(str)

    # Colors
    if f"{cluster_key}_colors" in dyn_adata.uns:
        cluster_colors = dyn_adata.uns[f"{cluster_key}_colors"]
        unique_clusters = dyn_adata.obs[cluster_key].cat.categories.astype(str)
    else:
        unique_clusters = clusters.unique()
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # Target cells
    target_cells = np.concatenate(
        [np.where(clusters == str(tc))[0] for tc in target_clusters]
    ) if len(target_clusters) else np.array([], dtype=int)
    if target_cells.size == 0:
        raise ValueError(f"No cells found in clusters {target_clusters}")

    # Walks that start in target cells
    valid_walks = [j for j in range(paths.shape[1]) if paths[0, j].item() in target_cells]
    if not valid_walks:
        raise ValueError(f"No walks start from clusters {target_clusters}")

    # Sample walks
    n_walks = min(n_walks, len(valid_walks))
    selected_walks = np.random.choice(valid_walks, size=n_walks, replace=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Anything with zorder > threshold will be rasterized
    if rasterize:
        ax.set_rasterization_zorder(rasterization_zorder)

    legend_handles, legend_labels = [], []

    # Background cells (low zorder → stay vector unless you want them rasterized explicitly)
    for i, cl in enumerate(unique_clusters):
        mask = clusters == cl
        color = cluster_colors[i] if isinstance(cluster_colors[i], str) else cluster_colors[i]
        sc = ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=10, alpha=0.6, label=f"{cl}",
            zorder=1  # <= below rasterization_zorder → stays vector
        )

        # If you want background points rasterized regardless, uncomment:
        if rasterize: sc.set_rasterized(True)
        legend_handles.append(sc)
        legend_labels.append(cl)

    # Walk trajectories (high zorder → will be rasterized when rasterize=True)
    walk_colors = plt.cm.plasma(np.linspace(0, 1, n_walks))
    for i, widx in enumerate(selected_walks):
        walk_path = paths[:, widx].detach().cpu().numpy()
        walk_xy = coords[walk_path]

        line = ax.plot(
            walk_xy[:, 0], walk_xy[:, 1],
            color=walk_colors[i], alpha=alpha_walk,
            linewidth=linewidth, zorder=10
        )[0]
        if rasterize: line.set_rasterized(True)

        # Start (circle) & End (square). Keep vector (low zorder) for crisp markers,
        # or bump zorder > threshold to rasterize them too.
        start = ax.scatter(
            walk_xy[0, 0], walk_xy[0, 1],
            color="black", s=80, marker="o",
            edgecolor="white", linewidth=1, zorder=1.5
        )
        end = ax.scatter(
            walk_xy[-1, 0], walk_xy[-1, 1],
            color="gold", s=80, marker="s",
            edgecolor="black", linewidth=1, zorder=1.5
        )
        # If you prefer these rasterized too:
        # if rasterize: 
        #     start.set_rasterized(True)
        #     end.set_rasterized(True)

    # Clean axes
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Sample Discretized Cell State Evolution", fontsize=14)

    # Legend (kept vector)
    ax.legend(
        legend_handles, legend_labels,
        bbox_to_anchor=(1.05, 1), loc="upper left",
        fancybox=True, shadow=True
    )

    plt.tight_layout()
    if filename is not None:
        # DPI controls the resolution of rasterized parts only (for vector formats)
        plt.savefig(filename, dpi=raster_dpi, bbox_inches="tight", transparent=True)
    plt.show()

    # Summary
    print("Visualization Summary:")
    print(f"- Total cells in dataset: {dyn_adata.n_obs}")
    print(f"- Cells in target clusters {target_clusters}: {len(target_cells)}")
    print(f"- Total walks available: {paths.shape[1]}")
    print(f"- Walks starting from target clusters: {len(valid_walks)}")
    print(f"- Walks visualized: {n_walks}")
    print(f"- Steps per walk: {paths.shape[0]}")




# def visualize_random_walks_on_umap(
#     dyn_adata, paths, target_clusters, 
#     cluster_key='clusters', n_walks=10, 
#     figsize=(12, 8), alpha_walk=0.7, 
#     linewidth=1.5, seed=42, rep="X_umap", filename=None
# ):
#     """
#     Visualize random walks on UMAP starting from cells in specific clusters.
#     """

#     # --- Validation ---
#     if rep not in dyn_adata.obsm:
#         raise KeyError(f"Embedding '{rep}' not found in dyn_adata.obsm")

#     if cluster_key not in dyn_adata.obs:
#         raise KeyError(f"Cluster key '{cluster_key}' not found in dyn_adata.obs")

#     # Set random seed for reproducibility
#     np.random.seed(seed)

#     # Extract embedding coordinates
#     coords = dyn_adata.obsm[rep]

#     # Ensure target_clusters is a list
#     if not isinstance(target_clusters, list):
#         target_clusters = [target_clusters]

#     # Get cluster information
#     clusters = dyn_adata.obs[cluster_key].astype(str)

#     # Get cluster colors from scanpy (if available)
#     if f"{cluster_key}_colors" in dyn_adata.uns:
#         cluster_colors = dyn_adata.uns[f"{cluster_key}_colors"]
#         unique_clusters = dyn_adata.obs[cluster_key].cat.categories.astype(str)
#     else:
#         unique_clusters = clusters.unique()
#         cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

#     # Find cells belonging to target clusters
#     target_cells = []
#     for target_cluster in target_clusters:
#         cluster_cells = np.where(clusters == str(target_cluster))[0]
#         target_cells.extend(cluster_cells)
#     target_cells = np.array(target_cells)

#     if len(target_cells) == 0:
#         raise ValueError(f"No cells found in clusters {target_clusters}")

#     # Find walks that start from target cluster cells
#     valid_walks = [
#         j for j in range(paths.shape[1]) 
#         if paths[0, j].item() in target_cells
#     ]

#     if len(valid_walks) == 0:
#         raise ValueError(f"No walks start from clusters {target_clusters}")

#     # Select random subset of valid walks
#     n_walks = min(n_walks, len(valid_walks))
#     selected_walks = np.random.choice(valid_walks, size=n_walks, replace=False)

#     # --- Create the plot ---
#     fig, ax = plt.subplots(figsize=figsize)

#     legend_handles = []
#     legend_labels = []

#     # Plot all cells as background using scanpy colors
#     for i, cluster in enumerate(unique_clusters):
#         cluster_mask = clusters == cluster
#         color = cluster_colors[i] if isinstance(cluster_colors[i], str) else cluster_colors[i]
#         scatter = ax.scatter(
#             coords[cluster_mask, 0], coords[cluster_mask, 1],
#             c=color, s=10, label=f"{cluster}", alpha=0.6
#         )
#         scatter.set_rasterized(True)  # rasterize background points
#         legend_handles.append(scatter)
#         legend_labels.append(cluster)

#     # Plot random walks
#     walk_colors = plt.cm.plasma(np.linspace(0, 1, n_walks))
#     for i, walk_idx in enumerate(selected_walks):
#         walk_path = paths[:, walk_idx].detach().cpu().numpy()  # GPU-safe
#         walk_coords = coords[walk_path]

#         # Line for walk
#         ax.plot(
#             walk_coords[:, 0], walk_coords[:, 1],
#             color=walk_colors[i], alpha=alpha_walk,
#             linewidth=linewidth, zorder=10
#         )

#         # Start point (black circle)
#         ax.scatter(
#             walk_coords[0, 0], walk_coords[0, 1],
#             color="black", s=80, marker="o",
#             edgecolor="white", linewidth=1, zorder=15
#         )

#         # End point (gold square)
#         ax.scatter(
#             walk_coords[-1, 0], walk_coords[-1, 1],
#             color="gold", s=80, marker="s",
#             edgecolor="black", linewidth=1, zorder=15
#         )

#     # Clean up axes
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     ax.set_title("Sample Discretized Cell State Evolution", fontsize=14)

#     # Legend
#     ax.legend(
#         legend_handles, legend_labels,
#         bbox_to_anchor=(1.05, 1), loc="upper left",
#         fancybox=True, shadow=True
#     )

#     plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.show()
    
#     # Print summary statistics
#     print(f"Visualization Summary:")
#     print(f"- Total cells in dataset: {dyn_adata.n_obs}")
#     print(f"- Cells in target clusters {target_clusters}: {len(target_cells)}")
#     print(f"- Total walks available: {paths.shape[1]}")
#     print(f"- Walks starting from target clusters: {len(valid_walks)}")
#     print(f"- Walks visualized: {n_walks}")
#     print(f"- Steps per walk: {paths.shape[0]}")


def plot_z_components(
    z_sol_subset,
    t_max,
    save_path: str = None,
    title: str = "ODE Solution Components in Cell State Space",
    subtitle: str = "(Selected Trajectories)",
    n_cols: int = 2,
    fig_size: tuple = (12, 14),
    cmap: str = "tab20",
    xlabel: str = "ODE Time Unit",
    ylabel: str = "Value",
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str = "white",
    edgecolor: str = "none",
):
    """
    Plot each latent component across multiple trajectories.

    Parameters
    ----------
    z_sol_subset : torch.Tensor or np.ndarray
        Array of shape (T, N, D): T time points, N trajectories, D components.
    t_max : float
        Upper bound of the time axis (lower bound is 0).
    save_path : str, optional
        If given, the figure will be saved to this path.
    title : str
        Supertitle for the figure.
    subtitle : str
        Subtitle (shown under the supertitle).
    n_cols : int
        Number of subplot columns (rows = ceil(D / n_cols)).
    fig_size : tuple
        Size of the entire figure (width, height).
    cmap : str
        A matplotlib colormap name for the trajectory colors.
    xlabel, ylabel : str
        Axis labels.
    dpi, bbox_inches, facecolor, edgecolor
        Passed to plt.savefig if saving.
    """
    # Convert to NumPy
    if torch.is_tensor(z_sol_subset):
        z_arr = z_sol_subset.cpu().numpy()
    else:
        z_arr = np.array(z_sol_subset)

    T, N, D = z_arr.shape
    t = np.linspace(0, t_max, T)

    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex='col')
    axes = axes.flatten()

    # prepare colors
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))

    for i in range(D):
        ax = axes[i]
        for j in range(N):
            ax.plot(t, z_arr[:, j, i],
                    color=colors[j],
                    alpha=1.0,
                    linewidth=2)
        ax.set_title(f'Component {i+1}', fontsize=14, pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)

        # only show x‑labels on bottom row
        row_idx = i // n_cols
        if row_idx == n_rows - 1:
            ax.set_xlabel(xlabel, fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelsize=10)

    # turn off any unused subplots
    for empty_ax in axes[D:]:
        fig.delaxes(empty_ax)

    # supertitle + subtitle
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.title(subtitle, fontsize=12, y=0.94)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # background
    fig.patch.set_facecolor(facecolor)

    # save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi,
                    bbox_inches=bbox_inches,
                    facecolor=facecolor,
                    edgecolor=edgecolor)
    plt.show()


def summary_scores(all_scores):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): 
            {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: 
            Group-wise aggregation scores.
        float: 
            score aggregated on all samples
        
    """
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg

def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): 
            Anndata object.
        nodes (list): 
            Indexes for cells
        target (str): 
            Cluster name.
        k_cluster (str): 
            Cluster key in adata.obs dataframe

    Returns:
        list: 
            Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]

def cross_boundary_correctness(
    adata, 
    k_cluster, 
    k_velocity, 
    cluster_edges, 
    return_raw=False, 
    x_emb="X_umap"
):
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        return_raw (bool): 
            return aggregated or raw scores.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    elif x_emb == "X_tsne":
        v_emb = adata.obsm['{}_tsne'.format(k_velocity)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    
    x_emb = adata.obsm[x_emb]
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
        
        boundary_nodes = [keep_type(adata, np.array(nodes), v, "clusters") for nodes in nbs]
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            type_score.append(np.mean(dir_scores))
        
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
        
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])

def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        return_raw (bool): 
            return aggregated or raw scores.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, k_cluster), nbs)

        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

def evaluate(
    adata, 
    cluster_edges, 
    k_cluster, 
    k_velocity="velocity", 
    x_emb="X_umap", 
    verbose=True
):
    """Evaluate velocity estimation results using 5 metrics.
    
    Args:
        adata (Anndata): 
            Anndata object.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            aggregated metric scores.
    
    """

    from utils import cross_boundary_correctness
    from utils import inner_cluster_coh
    crs_bdr_crc = cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, True, x_emb)
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, True)
    
    if verbose:
        print("# Cross-Boundary Direction Correctness (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(crs_bdr_crc)))
        print("# In-cluster Coherence\n{}\nTotal Mean: {}".format(*summary_scores(ic_coh)))
    
    return {
        "Cross-Boundary Direction Correctness (A->B)": crs_bdr_crc,
        "In-cluster Coherence": ic_coh,
    }
