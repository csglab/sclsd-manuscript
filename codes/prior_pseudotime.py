import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse as sp
import scanpy as sc
import torch
import warnings
import numpy as np
import pandas as pd
import sys
import pyro
from .utils import Prepare_simple_walks
from scipy.spatial.distance import cdist
import warnings
import torch
import copy
import random



    
def get_prior_pseudotime(data_dict, lsd, origin_cluster, k_cluster="clusters",
                         device=torch.device("cpu"), max_pseudotime_per_cluster=None):
    """
    Estimate a prior pseudotime for an AnnData object based on one or more origin clusters,
    scaled by a user-defined max pseudotime per cluster.

    Parameters
    ----------
    data_dict : dict
        Must contain "normal_counts" and "adata".
    lsd : object
        Trained LSD model with x_encoder and z_encoder.
    origin_cluster : str or list of str
        Cluster(s) in adata.obs[k_cluster] to treat as origin.
    k_cluster : str
        Name of column in adata.obs that contains cluster labels.
    device : torch.device
        Device for computation.
    max_pseudotime_per_cluster : dict or None
        Dictionary mapping cluster name -> maximum pseudotime (e.g. {"A": 5.0, "B": 3.2}).
        If None, defaults to 1.0 for all clusters.

    Returns
    -------
    temp_adata : AnnData
        Copy of input AnnData with new columns in .obs:
        - prior_pseudotime_<origin> for each origin,
        - prior_pseudotime (minimum across all origins).
    """
    if isinstance(origin_cluster, str):
        origin_clusters = [origin_cluster]
    else:
        origin_clusters = origin_cluster

    if max_pseudotime_per_cluster is None:
        max_pseudotime_per_cluster = {orig: 1.0 for orig in origin_clusters}

    temp_adata = data_dict["adata"].copy()

    # Get latent representations
    x = torch.from_numpy(data_dict["normal_counts"]).to(device)
    z_loc = lsd.x_encoder(x)[0].detach().cpu().numpy()
    B_loc, _ = lsd.z_encoder(lsd.x_encoder(x)[0])

    # Store latent representations
    temp_adata.obsm["diff_rep"] = B_loc.detach().cpu().numpy()
    temp_adata.obsm["latent_rep"] = z_loc

    diff_rep = temp_adata.obsm["diff_rep"]
    pseudotime_dict = {}

    for orig in origin_clusters:
        cluster_mask = temp_adata.obs[k_cluster] == orig
        cluster_cells = diff_rep[cluster_mask.values]
        other_cells = diff_rep[~cluster_mask.values]

        if cluster_cells.shape[0] == 0 or other_cells.shape[0] == 0:
            pseudotime = np.zeros(diff_rep.shape[0])
        else:
            distances = cdist(cluster_cells, other_cells, metric="euclidean")
            max_dist_idx = np.argmax(np.mean(distances, axis=1))
            cell_of_origin = cluster_cells[max_dist_idx]
            all_distances = np.linalg.norm(diff_rep - cell_of_origin, axis=1)

            if np.max(all_distances) == np.min(all_distances):
                normalized = np.zeros_like(all_distances)
            else:
                normalized = (all_distances - np.min(all_distances)) / (np.max(all_distances) - np.min(all_distances))

            scale = max_pseudotime_per_cluster.get(orig, 1.0)
            pseudotime = normalized * scale

        pseudotime_dict[orig] = pseudotime
        temp_adata.obs[f"prior_pseudotime_{orig}"] = pseudotime

    return temp_adata


def infer_prior_time(data_dict, device, pseudotime_cluster, n_epochs=20, random_state = 42):
    """
    Run the full pipeline to infer prior pseudotime on an AnnData object.
    
    This updated version is consistent with the revised get_prior_pseudotime function,
    allowing pseudotime to be computed with respect to one or more origin clusters.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    device : torch.device
        Device to run the model on.
    pseudotime_cluster : str or list of str
        Cluster label(s) to use in get_prior_pseudotime.
    n_trajetories : int, optional
        Number of trajectories for random walks.
    n_epochs : int, optional
        Number of training epochs.
    
    Returns
    -------
    AnnData
        AnnData object with inferred prior pseudotime added in .obs.
        When multiple origin clusters are provided, separate pseudotime columns
        are added (e.g. "prior_pseudotime_Fev+", etc.) and a combined "prior_pseudotime"
        is computed.
    """
    import warnings
    from .lsd_train import LSD
    from codes.config import LSDConfig
    SEED = random_state
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    pyro.set_rng_seed(SEED)
    
    # -----------------------------
    # Step 1: Preprocessing - Prepare the data dictionary.
    # -----------------------------

    
    # Use a copy of the processed AnnData for preparing random walks.
    temp = data_dict["adata"].copy()
    
    # Extract the connectivity matrix (assumed to be stored in .obsp["connectivities"])
    P = temp.obsp["connectivities"]
    # Normalize the sparse connectivity matrix to create transition probabilities
    row_sums = np.array(P.sum(axis=1)).flatten()
    P = sp.diags(1.0 / row_sums) @ P
    n_steps = 2
    n_trajectories = 2**(int(np.log2(len(temp))))  # Alternatively, you could use adata.n_obs // 2
    
    # Prepare random walks using GPU function (assumes your function is available)
    
    # -----------------------------
    # Step 2: Prepare the model
    # -----------------------------
    # Instantiate the custom activation function.
    cfg = LSDConfig()

    cfg.walks.batch_size = 256
    cfg.walks.path_len = 2
    cfg.walks.num_walks = 2**(int(np.log2(len(temp))))
    cfg.walks.random_state = SEED
    cfg.optimizer.wasserstein_schedule.max_W = 1
    cfg.optimizer.wasserstein_schedule.min_W = 1
    cfg.optimizer.adam.eta_min = 1e-4
    cfg.optimizer.adam.T_0 = 30
    cfg.model.z_dim = 50
    cfg.walks.batch_size = int(n_trajectories / 16)


    
    # Ignore UserWarnings for a cleaner output.
    warnings.filterwarnings("ignore", category=UserWarning)
    pyro.set_rng_seed(SEED)
    pyro.clear_param_store()
    temp_lsd = LSD(data_dict['adata'], cfg,device = device,
                   lib_size_key = "librarysize",raw_count_key="raw")
    temp_lsd.set_prior_transition(prior_transition = P)
    temp_lsd.prepare_walks(n_trajectories=n_trajectories)
    temp_lsd.train(num_epochs= n_epochs, plot_loss = False)
    # -----------------------------
    # Step 4: Infer prior pseudotime.
    # -----------------------------
    # This call now supports one or multiple origin clusters.
    temp_adata = get_prior_pseudotime(data_dict, temp_lsd.lsd, pseudotime_cluster, k_cluster="clusters", device=device)
    
    return temp_adata


def compute_normalized_cluster_connectivity(adata, cluster_key="clusters"):
    # Get connectivity matrix
    conn = adata.obsp["connectivities"]
    if sp.issparse(conn):
        conn = conn.tocsr()

    # Get cluster labels
    clusters = adata.obs[cluster_key].astype(str)
    cluster_ids = clusters.unique()
    cluster_to_idx = {cl: np.where(clusters == cl)[0] for cl in cluster_ids}

    # Initialize result matrix
    cluster_conn_matrix = pd.DataFrame(
        np.zeros((len(cluster_ids), len(cluster_ids))),
        index=cluster_ids,
        columns=cluster_ids,
    )

    # Fill in matrix with normalized connectivity
    for ci in cluster_ids:
        idx_i = cluster_to_idx[ci]
        for cj in cluster_ids:
            idx_j = cluster_to_idx[cj]
            sub_conn = conn[np.ix_(idx_i, idx_j)]
            total = sub_conn.sum()
            norm = len(idx_i) * len(idx_j)
            cluster_conn_matrix.loc[ci, cj] = total / norm if norm > 0 else 0
    cluster_conn_matrix /= cluster_conn_matrix.sum(axis = 0)
    for ci in cluster_ids:
         cluster_conn_matrix.loc[ci, ci] = 0

    return cluster_conn_matrix
def find_highly_connected_dict(cluster_conn, ratio = 0.1, hot_ratio = 0.05):
    # Copy to avoid mutating original
    conn = cluster_conn.copy()

    # Remove self-connections (set diagonal to NaN)
    np.fill_diagonal(conn.values, np.nan)




    # Initialize output dictionary
    high_conn_dict = {cluster: [] for cluster in conn.index}
    

    # Populate dictionary with eligible neighbors
    for source in conn.index:
        max_conn = np.nanmax(conn.loc[:,source].values)
        candidates = [conn.loc[source,:].idxmax()]
        high_conn_dict[source].append(candidates[0])
        threshold = max_conn * ratio
        for target in conn.columns:
            if source != target and conn.loc[source, target] >= threshold and target not in candidates:
                high_conn_dict[source].append(target)
                candidates.append(target)
        for target in conn.columns:
            if source != target and conn.loc[source, target] >= hot_ratio*max_conn and target not in candidates:
                if all(conn.loc[c, target] < conn.loc[source, target] for c in candidates):
                    
                    high_conn_dict[source].append(target)

    return high_conn_dict
def deduplicate_targets_by_connectivity(connectivity_dict, cluster_conn, clusters_of_interest):
    from collections import defaultdict

    # Create a mapping from target → list of source clusters that have it
    reverse_map = defaultdict(list)
    for source in clusters_of_interest:
        for target in connectivity_dict.get(source, []):
            reverse_map[target].append(source)

    # Resolve conflicts: for each target, keep only the best source
    final_dict = {source: [] for source in clusters_of_interest}

    for target, sources in reverse_map.items():
        if len(sources) == 1:
            final_dict[sources[0]].append(target)
        else:
            # Multiple sources → pick the one with highest connectivity to the target
            best_source = max(sources, key=lambda src: cluster_conn.loc[src, target])
            final_dict[best_source].append(target)

    return final_dict

def infer_phylo(adata, root, cluster_key="clusters", ratio=0.3, hot_ratio = 0.05):

    phylo_adata = adata.copy()
    all_clusters = set(phylo_adata.obs[cluster_key].unique())
    assigned_clusters = set([root])
    assinged_origins = set([root])
    phylo = {root: []}
    origins = [root]

    while len(assigned_clusters) < len(all_clusters) and len(origins) > 0:
        # Compute cluster-cluster connectivity
        conn = compute_normalized_cluster_connectivity(phylo_adata, cluster_key=cluster_key)
        conn_dict = find_highly_connected_dict(conn, ratio=ratio, hot_ratio = hot_ratio)

        # Prevent any origin from being assigned to another cluster
        for k in conn_dict:
            conn_dict[k] = [t for t in conn_dict[k] if t not in origins]

        # Deduplicate connections to ensure unique assignment
        conn_dict = deduplicate_targets_by_connectivity(conn_dict, conn, origins)
        # Update phylo and assigned clusters
        new_origins = []
        for origin in origins:
            children = conn_dict.get(origin, [])
            phylo[origin] = children
            new_origins.extend(children)
            assigned_clusters.update(children)
            assinged_origins.update(origin)

        # Remove newly assigned clusters from phylo_adata
        phylo_adata = phylo_adata[~phylo_adata.obs[cluster_key].isin(origins)]

        # Recompute neighborhood graph for remaining data
        if phylo_adata.n_obs > 0:
            sc.pp.pca(phylo_adata)
            sc.pp.neighbors(phylo_adata)

        origins = new_origins

    # Assign unassigned clusters to themselves (unreachable clusters)
    unassigned = all_clusters - assigned_clusters
    for cluster in unassigned:
        phylo[cluster] = []
    for cluster in all_clusters:
        if cluster not in phylo:
            phylo[cluster] = []

    return phylo

def get_tree_branches(phylo, root):
    branches = []

    def dfs(node, path):
        children = phylo.get(node, [])
        if not children:  # Leaf node
            branches.append(path + [node])
        else:
            for child in children:
                dfs(child, path + [node])

    dfs(root, [])
    return branches
def infer_global_pseudotime(adata, phylo, root, device, cluster_key="clusters", n_epochs=20, random_state = 42):
    main_adata = adata.copy()
    branches = get_tree_branches(phylo, root)
    cols = []
    for branch in branches:

        # Select only cells in the current branch
        branch_adata = adata[adata.obs[cluster_key].isin(branch)].copy()
        if "X_pca" in branch_adata.obsm:
            del branch_adata.obsm["X_pca"]
        if "connectivities" in branch_adata.obsp:
            del branch_adata.obsp["connectivities"]
        data_dict = Prepare_DataDict(
            adata, 
            n_top_genes=None, 
            Normalize= False,
            target_sum=1e4, 
        )

        root_cell = branch[0]

        # Run pseudotime inference
        branch_pseudotime_adata = infer_prior_time(data_dict, device, root_cell, n_epochs=n_epochs,random_state = random_state)

        main_adata.obs[f"prior_pseudotime_{branch[-1]}"] = None
        main_adata.obs.loc[adata.obs[cluster_key].isin(branch), f"prior_pseudotime_{branch[-1]}"] = \
        branch_pseudotime_adata.obs[f"prior_pseudotime_{root}"]
        cols.append(f"prior_pseudotime_{branch[-1]}")

    
    main_adata.obs["prior_pseudotime"] = main_adata.obs[cols].mean(axis=1, skipna=True)
    main_adata = shift_child_pseudotime(main_adata, phylo, cluster_key=cluster_key, pseudotime_key="prior_pseudotime")
    pseudotime = main_adata.obs["prior_pseudotime"]
    pseudotime = np.array(pseudotime, dtype=float)
    min_val = pseudotime.min()
    max_val = pseudotime.max()
    main_adata.obs["prior_pseudotime"] = (pseudotime - min_val) / (max_val - min_val)
    return main_adata

def shift_child_pseudotime(temp, phylo, cluster_key="clusters", pseudotime_key="prior_pseudotime_mean"):
    visited = set()

    def adjust_children(parent):
        parent_mask = temp.obs[cluster_key] == parent
        parent_max = temp.obs.loc[parent_mask, pseudotime_key].max()

        for child in phylo.get(parent, []):
            if child in visited:
                continue

            child_mask = temp.obs[cluster_key] == child
            child_min = temp.obs.loc[child_mask, pseudotime_key].min()

            # Compute shift if necessary
            K = parent_max - child_min
            shift = np.maximum(0, K)

            if shift > 0:
                temp.obs.loc[child_mask, pseudotime_key] += shift

            visited.add(child)
            adjust_children(child)  # Recurse

    roots = [k for k in phylo if all(k not in v for v in phylo.values())]
    for root in roots:
        adjust_children(root)

    return temp




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

def random_walks_gpu(T, n_steps, n_trajectories):
    """
    Generate random walks in parallel on GPU.
    
    Parameters:
        T (torch.Tensor): Dense transition probability matrix of shape (n_cells, n_cells) on GPU.
        n_steps (int): Number of steps per walk.
        n_trajectories (int): Number of random walks to generate.
    
    Returns:
        torch.Tensor: Tensor of shape (n_trajectories, n_steps+1) containing the cell indices visited.
    """
    n_cells = T.shape[0]
    
    # Initialize all walks with a random starting cell for each trajectory
    current_states = torch.randint(0, n_cells, (n_trajectories,), device="cuda", dtype=torch.long)
    walks = torch.empty((n_trajectories, n_steps), dtype=torch.long, device="cuda")
    walks[:, 0] = current_states
    
    # Simulate walks in parallel (vectorized across trajectories)
    for step in range(1, n_steps):
        # For each trajectory, sample the next cell using its row probabilities.
        # T[current_states] gives a tensor of shape (n_trajectories, n_cells)
        next_states = torch.multinomial(T[current_states], num_samples=1).squeeze(1)
        walks[:, step] = next_states
        current_states = next_states
        
    return walks

def Prepare_walks_gpu(n_steps, n_trajectories, connectivity):
    """
    Prepares multiple random walks using GPU acceleration.
    
    Parameters:
        n_steps (int): Number of steps per walk.
        n_trajectories (int): Number of random walks to generate.
        connectivity (scipy.sparse matrix): Sparse connectivity matrix.
    
    Returns:
        torch.Tensor: Tensor of shape (n_trajectories, n_steps+1) with random walk trajectories on GPU.
    """
    # Convert connectivity matrix to a dense transition matrix on GPU
    T_gpu = prepare_transition_matrix_gpu(connectivity)
    # Generate the random walks in parallel on GPU
    walks = random_walks_gpu(T_gpu, n_steps, n_trajectories)
    walks = walks.cpu()
    return walks

def get_all_descendants(cluster, phylogeny, descendants=None):
    """
    Recursively find all descendants of a cluster in the phylogeny
    
    Parameters:
    -----------
    cluster : str
        The cluster to find descendants for
    phylogeny : dict
        Dictionary with format {parent: [child1, child2, ...]}
    descendants : set, optional
        Set to store descendants (used in recursive calls)
        
    Returns:
    --------
    descendants : set
        Set of all descendant clusters
    """
    if descendants is None:
        descendants = set()
    
    # Add direct children
    direct_children = phylogeny.get(cluster, [])
    descendants.update(direct_children)
    
    # Recursively add children of children
    # for child in direct_children:
    #     descendants.update(get_all_descendants(child, phylogeny))
    
    return descendants

def create_phylogeny_matrix(adata, phylogeny):
    """
    Create a cell-cell matrix where aij = 1 if:
    - cells i and j belong to the same cluster, or
    - cell j belongs to any descendant cluster of cell i's cluster
    
    Parameters:
    -----------
    adata : AnnData object
        The annotated data matrix with cluster annotations
    phylogeny : dict
        Dictionary with format {parent: [child1, child2, ...]}
    
    Returns:
    --------
    phylo_matrix : scipy.sparse.csr_matrix
        Cell-cell matrix based on phylogeny relationships
    """
    # Extract cluster information
    clusters = adata.obs['clusters'].unique().tolist()  # Assuming cluster info is in adata.obs['cluster']
    
    # Create a dictionary mapping cells to their cluster
    cell_to_cluster = dict(zip(adata.obs_names, adata.obs['clusters']))
    
    # Precompute all descendants for each cluster to avoid repeated computation
    all_descendants = {}
    for cluster in clusters:
        all_descendants[cluster] = get_all_descendants(cluster, phylogeny)
    
    # Initialize a matrix of zeros with shape (n_cells, n_cells)
    n_cells = adata.shape[0]
    phylo_matrix = np.zeros((n_cells, n_cells))
    
    # Fill the matrix based on the rules
    for i, cell_i in enumerate(adata.obs_names):
        cluster_i = cell_to_cluster[cell_i]
        
        for j, cell_j in enumerate(adata.obs_names):
            cluster_j = cell_to_cluster[cell_j]
            
            # Set aij = 1 if cells i and j belong to the same cluster
            if cluster_i == cluster_j:
                phylo_matrix[i, j] = 1
            
            # Set aij = 1 if cell j's cluster is any descendant of cell i's cluster
            elif cluster_j in all_descendants.get(cluster_i, set()):
                phylo_matrix[i, j] = 1
    
    # Convert to sparse matrix for efficiency
    phylo_matrix_sparse = sp.csr_matrix(phylo_matrix)
    
    return phylo_matrix_sparse


def prior_transition_matrix(adata, timekey, beta_t, obsp_key = "connectivities"):
    """
    Build a transition matrix from an AnnData object using pseudotime and a representation.

    The function assumes:
      - The connectivity matrix is stored in adata.obsp["connectivities"].
      - Pseudotime is stored in adata.obs[timekey].

    The transition probability from cell i to cell j is computed as:
        p_ij = exp(beta_t * (t_j - t_i)) / Z_i,
    where:
        - t_i and t_j are pseudotime values,
        - Z_i is a normalization constant for cell i.

    Parameters:
    -----------
    adata : AnnData
        The annotated data object.
    timekey : str
        The key in adata.obs where pseudotime values are stored.
    beta_t : float
        Scaling parameter for the pseudotime differences.

    Returns:
    --------
    transition : scipy.sparse.csr_matrix
        A sparse transition matrix of shape (n_cells, n_cells) with normalized transition probabilities.
    """
    # Extract the required data from adata.
    connectivity = adata.obsp[obsp_key].tocsr()
    pseudotime = adata.obs[timekey].values
    n_cells = connectivity.shape[0]

    row_idx, col_idx, data_vals = [], [], []
    
    # Iterate over each cell to compute its transition probabilities.
    for i in range(n_cells):
        # Find neighbors for cell i
        start = connectivity.indptr[i]
        end = connectivity.indptr[i+1]
        neighbors = connectivity.indices[start:end]
        
        if len(neighbors) == 0:
            continue
        
        # Pseudotime differences: t_j - t_i for each neighbor j.
        dt = pseudotime[neighbors] - pseudotime[i]
        # Compute Euclidean distances between cell i and its neighbors.
        
        # Compute unnormalized weights
        weights = np.exp(beta_t * dt)
        
        # Normalize the weights to sum to 1
        Z = weights.sum()
        normalized_weights = weights / Z if Z > 0 else np.zeros_like(weights)
        
        # Record the computed probabilities.
        row_idx.extend([i] * len(neighbors))
        col_idx.extend(neighbors)
        data_vals.extend(normalized_weights)
    
    # Construct the sparse transition matrix.
    transition = sp.csr_matrix((data_vals, (row_idx, col_idx)), shape=(n_cells, n_cells))
    return transition



def Prepare_DataDict(adata, n_top_genes=5000, target_sum=1e4, Normalize = True, log=True, n_pcs=50, 
                     use_rep="X_pca", n_neighbors=15, gene_selection="hvg"):
    """
    Prepares a dictionary of processed data from an AnnData object.

    Parameters:
    ----------
    adata : AnnData
        The input single-cell dataset.
    min_counts : int
        Minimum counts required for a gene to be kept.
    n_top_genes : int
        Number of highly variable genes to retain (if `gene_selection="hvg"`).
    min_cell_counts : int
        Minimum counts required for a cell to be kept.
    min_cell_genes : int
        Minimum number of genes expressed in a cell to be kept.
    target_sum : float or None
        Target sum for total counts normalization.
    log : bool
        Whether to apply log1p transformation.
    n_pcs : int
        Number of principal components to compute.
    use_rep : str
        Key in `adata.obsm` to use as the representation for neighbors.
    n_neighbors : int
        Number of neighbors for graph construction.
    gene_selection : str
        Choose "hvg" for highly variable genes or "custom" to use genes from `adata.uns["selected_genes"]`.

    Returns:
    -------
    dict
        A dictionary containing raw counts, normalized counts, library size, and processed AnnData.
    """
    
    adata = adata.copy()
    adata.raw = adata
    


    if gene_selection == "hvg":
        if n_top_genes != None:
            # Select highly variable genes
            sc.pp.filter_genes_dispersion(adata, n_top_genes=n_top_genes)

    elif gene_selection == "custom":
        if "selected_genes" in adata.uns:
            selected_genes = list(set(adata.uns["selected_genes"]) & set(adata.var.index))
            if len(selected_genes) == 0:
                raise ValueError("No matching genes found in adata.var.index for the custom gene list.")
            adata = adata[:, selected_genes].copy()
        else:
            raise KeyError("No 'selected_genes' found in adata.uns.")



    # Keep raw counts
    raw_counts = adata.X.copy()

    # Compute library size
    adata.obs['librarysize'] = adata.X.sum(axis=1)
    if Normalize:
        # Normalize counts
        sc.pp.normalize_total(adata, target_sum=target_sum)

        # Log transformation (if needed)
        if log:
            sc.pp.log1p(adata)

    # Compute PCA if not already present
    if "X_pca" not in adata.obsm.keys():
        sc.pp.pca(adata, n_comps=n_pcs)

    # Compute neighbors if not already present
    if "connectivities" not in adata.obsp.keys():
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    
    row_sums = adata.obsp["connectivities"].sum(axis=1).A1  # Convert sparse matrix to array

    # Identify valid cells (nonzero row sum)
    valid_cells = row_sums > 0
    if len(adata)- len(adata[valid_cells]) != 0:
        print(f"Cells with zero connectivity row sum: {len(adata)- len(adata[valid_cells])}")
        # Filter the cells
        adata = adata[valid_cells].copy()



    # Prepare output dictionary
    data_dict = {
        "raw_counts": raw_counts.toarray(),
        "normal_counts": adata.X.copy().toarray(),
        "librarysize": adata.obs["librarysize"].copy().values,
        "adata": adata.copy(),
    }

    return data_dict

def get_prior_transition(adata, n_steps, n_trajectories = None, device= torch.device("cpu"),cluster_key = "clusters",
                             phylogeny = None,root_cluster = None, time_key = None, random_state = 42, 
                               beta_t = 50, ratio = 0.5, hot_ratio = 0.05, plot = False, rep = None):

    adata = adata.copy()
    if time_key != None:
        if phylogeny != None:
            A = create_phylogeny_matrix(adata, phylogeny)
            connectivity = adata.obsp['connectivities']
            adata.obsp['phylogeny_conn'] = connectivity.multiply(A)
            adata.obsp['prior_transition'] = prior_transition_matrix(adata, time_key, beta_t, obsp_key = "phylogeny_conn")

        else: 
            adata.obsp['prior_transition'] = prior_transition_matrix(adata, time_key, beta_t, obsp_key = "connectivities")
    else:
        if phylogeny != None:
            A = create_phylogeny_matrix(adata, phylogeny)
            connectivity = adata.obsp['connectivities']
            A_masked = connectivity.multiply(A)
            adata.obsp['phylogeny_conn'] = A_masked
            row_sums = np.array(A_masked.sum(axis=1)).ravel()
            valid_cells = row_sums > 0

            # --- 3) remove any “dead” cells ---
            n_dead = np.sum(~valid_cells)
            if n_dead > 0:
                print(f"[LSD] Removing {n_dead} cells with no transitions")
                # filter your AnnData:
                #   this returns a new AnnData with only valid_cells
                adata = adata[valid_cells]
            root = list(phylogeny.keys())[0]
            temp = infer_global_pseudotime(adata, phylogeny, root, device, cluster_key="clusters", n_epochs=5, random_state = random_state)
            adata.obs["prior_pseudotime"] = temp.obs["prior_pseudotime"]
            adata.obsp['prior_transition'] = prior_transition_matrix(adata, "prior_pseudotime", beta_t, obsp_key = "phylogeny_conn")
        else:
            if root_cluster is None:
                raise ValueError("You should specify the root cluster.")

            phylo = infer_phylo(adata, root_cluster, cluster_key="clusters", ratio= ratio, hot_ratio = hot_ratio)
            A = create_phylogeny_matrix(adata, phylo)
            connectivity = adata.obsp['connectivities']
            adata.obsp['phylogeny_conn'] = connectivity.multiply(A)
            temp = infer_global_pseudotime(adata, phylo, root_cluster, device, cluster_key="clusters", n_epochs=5, random_state = random_state)
            adata.obs["prior_pseudotime"] = temp.obs["prior_pseudotime"]
            adata.obsp['prior_transition'] = prior_transition_matrix(adata, "prior_pseudotime", beta_t, obsp_key = "phylogeny_conn")

    return adata, adata.obsp['prior_transition'].toarray()
