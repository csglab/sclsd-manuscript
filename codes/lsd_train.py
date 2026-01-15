import scanpy as sc
import os
import numpy as np
import matplotlib as mpl
import matplotlib.collections as mcoll
from math import log2
from scipy.spatial.distance import cdist
import scipy.sparse as sp
import torch
import torch.nn as nn
import pyro
from torch.optim import Adam
from pyro.optim import CosineAnnealingWarmRestarts
from pyro.infer import SVI, TraceGraph_ELBO, TraceEnum_ELBO
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
from cellrank.kernels import ConnectivityKernel
from torchdiffeq import odeint
from dataclasses import replace
from typing import Optional, Dict

from .model import LSD_Model
from .config import LSDConfig, WalkConfig

class LSD:
    """
    Class for Latent State Dynamics (LSD) modeling and training.

    Attributes:
        adata (anndata.AnnData): Preprocessed single-cell AnnData object. Should contain (log)normalized counts in `adata.X`.
        layer_dims (dict): Dictionary specifying the architecture (layer dimensions and potential activation function) for network components.
        z_dim (int): Latent cell state dimension.
        B_dim (int): Differentiation state dimension.
        batch_size (int): Batch size.
        path_len (int): Length of each trajectory (random walk).
        device (torch.device): Training device (cpu or cuda).
        lib_size_key (str): Key for library size column in `AnnData.obs`.
        raw_count_key (str): Key for raw count data in `AnnData.layers`.
        optim_args (dict, optional): Adam optimizer and scheduler hyperparameters.
        KL_args (dict, optional): Annealing hyperparameters for the KL term.
        W_args (dict, optional): Wasserstein regularization schedule.
        V_coeff (float, optional): Regularization coefficient for the Waddington potential.
    """
    def __init__(
        self,
        adata,
        config: Optional[LSDConfig] = None,
        *,
        device: torch.device = torch.device("cuda"),
        lib_size_key: str = "librarysize",
        raw_count_key: str = "raw",
    ):


            self.config = config if isinstance(config, LSDConfig) else None
            if self.config is not None:
                model_cfg = self.config.model
                walk_cfg = replace(self.config.walks)
                opt_cfg = self.config.optimizer

                layer_dims = model_cfg.layer_dims.as_dict()
                z_dim = model_cfg.z_dim
                B_dim = model_cfg.B_dim
                batch_size = walk_cfg.batch_size
                path_len = walk_cfg.path_len
                walk_cfg.batch_size = batch_size
                walk_cfg.path_len = path_len
                V_coeff = model_cfg.V_coeff
                self.optim_args = opt_cfg.adam.as_dict()
                self.KL_args = opt_cfg.kl_schedule.as_dict()
                self.W_args = opt_cfg.wasserstein_schedule.as_dict()
            else:
                if layer_dims is None:
                    raise ValueError("layer_dims must be provided when no LSDConfig is passed.")
                z_dim = z_dim if z_dim is not None else 10
                B_dim = B_dim if B_dim is not None else 2
                batch_size = batch_size if batch_size is not None else 256
                path_len = path_len if path_len is not None else 16
                V_coeff = 0 if V_coeff is None else V_coeff

            self.walk_config = (
                walk_cfg
                if self.config is not None
                else WalkConfig(batch_size=batch_size, path_len=path_len)
            )
            #adata params
            num_genes = len(adata.var)
            lib_size = adata.obs[lib_size_key]
            # Convert to float for PyTorch/Pyro compatibility
            self.xl_loc = float(lib_size.mean())
            self.xl_scale = float(lib_size.std())

            self.adata = adata.copy()
            self.lib_size_key = lib_size_key
            self.raw_count_key = raw_count_key
            self.cluster_key = None

            # Model params
            self.batch_size = batch_size
            self.z_dim = z_dim
            self.B_dim = B_dim
            self.device = device
            self.epoch = 0
            self.path_len = path_len
            # Initialization params
            self.P = None
            self.walks = None
            self.phylogeny = None

            # Cell fates
            self.paths = None
            self.z_sol = None
            self.fates = None



            self.lsd = LSD_Model(B_dim = B_dim, z_dim= z_dim, num_genes = num_genes,
                                layer_dims= layer_dims, batch_size = batch_size, path_len = path_len,device = device,
                                scale_factor=1.0 / (batch_size * path_len* num_genes), V_coeff = V_coeff,
                                xl_loc = self.xl_loc, xl_scale = self.xl_scale)
            self.lsd.to(self.device)
            


        
    def prepare_datadict(self):
        """
        Prepare a dictionary with arrays and metadata needed for training.
        """
        data_dict = {
            "raw_counts": self.adata.layers[self.raw_count_key].toarray(),
            "normal_counts": self.adata.X.copy().toarray(),
            "librarysize": self.adata.obs[self.lib_size_key].copy().values,
            "adata": self.adata.copy(),
            }
        return data_dict
    

    def _calculate_annealing_factor(self):
        """
        Annealing factor schedule for KL term.
        """
        min_af = self.KL_args['min_af']
        max_af = self.KL_args['max_af']
        max_epoch = self.KL_args['max_epoch']
        if self.epoch < max_epoch:
            af = min_af + (max_af - min_af) * self.epoch / max_epoch
        else:
            af = max_af
        return af


    def _calculate_W_factor(self):
        """
        Schedule for Wasserstein regularization factor.
        """
        min_W = self.W_args['min_W']
        max_W = self.W_args['max_W']
        max_epoch = self.W_args['max_epoch']
        if self.epoch < max_epoch:
            W = max_W + (min_W - max_W) * self.epoch / max_epoch
        else:
            W = min_W
        return W

    def _V(self, x):
        """
        For monitoring Potential decreasing
        """
        self.lsd.eval()
        with torch.no_grad():
            loc, _ = self.lsd.x_encoder(x)
            V = self.lsd.potential(loc)
            dim = V.shape[-1]
            V = V.reshape(int(len(x)/self.path_len), self.path_len, dim)
            S = 0
            for i in range(1,self.path_len):
                term1 = V[:,i,:]
                term2 = V[:,i-1,:]
                gate = F.relu(term1- term2)
                S += torch.mean(gate)/self.path_len
            
            
        return S.detach().cpu().numpy()
    def _H(self, x):
        """
        For monitoring Potential decreasing
        """
        self.lsd.eval()
        with torch.no_grad():
            loc, _ = self.lsd.x_encoder(x)
            _ , sigma = self.lsd.z_encoder(loc)
            dim = sigma.shape[-1]
            sigma = sigma.reshape(int(len(x)/self.path_len), self.path_len, dim)
            S = 0
            for i in range(1,self.path_len):
                term1 = 0.5* torch.log(sigma[:,i,:]**2).sum(axis= -1)
                term2 = 0.5* torch.log(sigma[:,i-1,:]**2).sum(axis= -1)
                gate = F.relu(term1- term2)
                S += torch.mean(gate)/self.path_len
            
            
        return S.detach().cpu().numpy()
    
    def z_rec_loss(self, x):
        """
        For monitoring Reconstruction of z from B
        """
        self.lsd.eval()
        with torch.no_grad():

            z, _ = self.lsd.x_encoder(x)
            b , _ = self.lsd.z_encoder(z)
            z_hat , var = self.lsd.B_decoder(b)
            loss = torch.norm((z_hat - z), p = 2 , dim = -1)
        return torch.mean(loss)


    
    def prepare_dataset(self, walks, data_dict):

        """
        This function Prepares batches from a set of random walks
        """
        x = torch.from_numpy(data_dict["normal_counts"]).type(torch.float32)
        x_raw = torch.from_numpy(data_dict["raw_counts"]).type(torch.float32)
        xl = torch.from_numpy(data_dict["librarysize"]).unsqueeze(-1)
        x = x[walks]
        x_raw = x_raw[walks]
        xl = xl[walks].type(torch.float32)
        dataset = TensorDataset(x_raw, x, xl)        
        return dataset
    

    def train(self, num_epochs = 60,
              save_dir = None,save_interval=50, plot_loss = True, random_state: Optional[int] = None):
        """
        Train the LSD model on given AnnData and walk indices.
        Tracks and plots ELBO and V metrics.

        Args:
            adata (AnnData): Single-cell data object.
            walks (ndarray): Index array for synthetic or empirical random walks.
            num_epochs (int): Number of training epochs.
            raw_count_key (str): Layer name for raw counts.
        """
        data_dict = self.prepare_datadict()
        adata = data_dict["adata"].copy()
        if save_dir is not None:

            adata_dir = save_dir + "/adata"
            os.makedirs(adata_dir, exist_ok=True)
            adata_name = "training_adata.h5ad"
            adata.write(os.path.join(adata_dir, adata_name))

        if random_state is None:
            random_state = getattr(self.walk_config, "random_state", 42)

        # pyro.clear_param_store()
        pyro.set_rng_seed(random_state)
        pyro.enable_validation(True)
        self.lsd = self.lsd.to(self.device)
        scheduler = CosineAnnealingWarmRestarts({'optimizer': Adam,
                                                'optim_args': {'lr': self.optim_args['lr']},
                                                'T_0': self.optim_args['T_0'],
                                                'eta_min': self.optim_args['eta_min'],'T_mult': self.optim_args['T_mult']},
                                                {"clip_norm": 10.0})
        elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
        svi = SVI(self.lsd.model, self.lsd.guide, scheduler, elbo)
        ELBO_losses = []
        V_losses = []
        H_losses = []
        epoch_bar = trange(num_epochs, desc="Training Epochs")
        for epoch in epoch_bar:
            epoch_losses = []
            V_loss = []
            H_loss = []
            #Shuffling the random walks for each epoch
            shuffled_walks = self.walks[torch.randperm(self.walks.size(0))]
            dataset = self.prepare_dataset(shuffled_walks, data_dict)

            for i in tqdm(range(0, len(dataset), self.batch_size),
                      desc=f"Epoch {epoch}", leave=False):
                self.lsd.train()
                batch = dataset[i:(i + self.batch_size)] 
                
                x_raw, x, xl = batch
                x_raw, x, xl = x_raw.to(self.device), x.to(self.device), xl.to(self.device).unsqueeze(-1).reshape(-1,1)
                batch_size, path_len, x_dim = x.shape
                assert self.path_len == path_len, (
                    f"path_len mismatch: LSD was initialized with path_len={self.path_len}, "
                    f"but got path_len={path_len} in the dataloader."
                    )
                x_raw, x = x_raw.reshape(-1, x_dim), x.reshape(-1, x_dim)
                annealing_factor = self._calculate_annealing_factor()
                W_coeff = self._calculate_W_factor()
                loss = svi.step(x_raw, x, xl = None, annealing_factor = annealing_factor, W_coeff = W_coeff)
                epoch_losses.append(loss)
                V_loss.append(self._V(x))
                H_loss.append(self._H(x))
                epoch_bar.set_postfix({"-ELBO": f"{epoch_losses[-1]:.4f}"})
                epoch_bar.set_postfix({"-ELBO": f"{epoch_losses[-1]:.4f}", "Potential Metric": f"{V_loss[-1]:.4f}", "Entropy Metric": f"{H_loss[-1]:.4f}",})

            epoch_loss_mean = np.mean(epoch_losses)
            ELBO_losses.append(epoch_loss_mean)
            V_loss_mean = np.mean(V_loss)
            V_losses.append(V_loss_mean)
            H_loss_mean = np.mean(H_loss)
            H_losses.append(H_loss_mean)

            if save_dir is not None:
                if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                    checkpoint_name = f"lsd_model_epoch{epoch+1:04d}.pth"
                    self.save(dir_path=save_dir, file_name=checkpoint_name)
            self.epoch+= 1
            

        
        if plot_loss:
            fig, axs = plt.subplots(1, 3, figsize=(12, 5))
            # Plot ELBO loss
            axs[0].plot(ELBO_losses, label="ELBO Loss")
            axs[0].set_title("ELBO Loss over Epochs")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("ELBO Loss")
            axs[0].legend()
            axs[0].grid(True)
            
            # Plot V loss
            axs[1].plot(V_losses, label="V Loss", color='orange')
            axs[1].set_title("V Loss over Epochs")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("V Loss")
            axs[1].legend()
            axs[1].grid(True)

            # Plot H loss
            axs[2].plot(H_losses, label="H Loss", color='red')
            axs[2].set_title("H difference over Epochs")
            axs[2].set_xlabel("Epoch")
            axs[2].set_ylabel("H difference")
            axs[2].legend()
            axs[2].grid(True)
            
            plt.tight_layout()
            out_path = os.path.join(save_dir, "loss_curves.png")
            fig.savefig(out_path)
            plt.show()


    def save(self, dir_path="checkpoints", file_name="model_and_params.pth"):
        """
        Saves the LSD model's state_dict and Pyro's parameter store to a file
        in the specified directory.

        Args:
            dir_path (str): Directory to save the checkpoint.
            file_name (str): Filename for the checkpoint.
        """
        os.makedirs(dir_path, exist_ok=True)  # create directory if it doesn't exist
        file_path = os.path.join(dir_path, file_name)
        torch.save({
            'model_state_dict': self.lsd.state_dict(),
            'pyro_params': pyro.get_param_store().get_state(),
        }, file_path)



    def load(self, dir_path="checkpoints", file_name="model_and_params.pth"):
        """
        Loads the LSD model's state_dict and Pyro's parameter store from a file
        in the specified directory.

        Args:
            dir_path (str): Directory to load the checkpoint from.
            file_name (str): Filename for the checkpoint.

        Returns:
            self: Returns self for chaining.
        """
        file_path = os.path.join(dir_path, file_name)
        checkpoint = torch.load(file_path, map_location=self.device)

        # Load model state_dict
        self.lsd.load_state_dict(checkpoint['model_state_dict'])

        # Load Pyro parameter store
        pyro.get_param_store().set_state(checkpoint['pyro_params'])

        print(f"[LSD] Model and Pyro parameters loaded from {file_path}")
        return self
    

    def get_variables(self, x):
        """
        Compute and return key latent variables and metrics after training.

        Args:
            x (Tensor): Input data.

        Returns:
            B_loc (Tensor): Differentiation state representation.
            z_loc (Tensor): Cell state representation.
            entropy (Tensor): Entropy of cell state (log-scale sum of std).
            potential (Tensor): Potential value for each cell.
            pseudotime (Tensor): Normalized pseudotime based on potential.
        """
        x = x.to(self.device)
        self.lsd.eval()
        with torch.no_grad():
            # Get latent representation of cell state
            z_loc, _ =  self.lsd.x_encoder(x)
            # Get differentiation state (mean and std)
            B_loc, B_scale = self.lsd.z_encoder(z_loc)
            # Entropy (sum log of scales)
            entropy = torch.log(B_scale).sum(dim=1)
            # Potential energy
            potential = self.lsd.potential(z_loc)
            # Pseudotime: normalized difference from max potential
            max_potential = torch.max(potential)
            pseudotime = max_potential - potential
            pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-8)
        return B_loc, z_loc, entropy, potential, pseudotime
    def calculate_transition_probs(self, potential, connectivity_matrix, beta=1.0):
        """
        Computes cell-cell transition probability matrix using Boltzmann weights, with binary connectivities.
        Args:
            potential (np.ndarray): shape (N,)
            connectivity_matrix (np.ndarray): shape (N, N), binary (0/1)
            beta (float): Boltzmann scaling factor

        Returns:
            np.ndarray: Transition probability matrix (N, N), row-normalized.
        """
        # Ensure correct types
        potential = potential.astype(float)

        # Step 1: Compute energy differences for each i,j (broadcasting)
        energy_diff = potential[None, :] - potential[:, None]  # shape (N, N)
        boltzmann_weights = np.exp(-beta * energy_diff)

        # Step 2: Apply connectivity mask
        boltzmann_weights *= connectivity_matrix

        # Step 3: Row-normalize
        row_sums = boltzmann_weights.sum(axis=1, keepdims=True) + 1e-12
        transition_matrix = boltzmann_weights / row_sums

        return transition_matrix

    
    def set_adata(self, adata):
        self.adata = adata

    def get_adata(self):
        """
        Postprocesses and annotates the adata with LSD results:
        - Adds B_loc, z_loc, entropy, potential, pseudotime to adata.obs
        - Computes and adds transition probability matrix to adata.obsp['transitions']

        Returns:
            AnnData: The annotated AnnData object.
        """
        # Copy adata to avoid side effects
        adata = self.adata.copy()
        # Get numpy array of input data
        x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        # Calculate latent and derived variables
        B_loc, z_loc, entropy, potential, pseudotime = self.get_variables(torch.from_numpy(x).float())

        adata.obsm["X_cell_state"] = z_loc.cpu().numpy()
        adata.obsm["X_diff_state"] = B_loc.cpu().numpy()
        adata.obs['entropy'] = entropy.cpu().numpy()
        adata.obs['potential'] = potential.cpu().numpy()
        adata.obs['lsd_pseudotime'] = pseudotime.cpu().numpy()

        # Check for connectivities
        if "connectivities" not in adata.obsp:
            raise KeyError(
                "'connectivities' matrix not found in adata.obsp. "
                "Run neighbors graph computation."
            )
        connectivity = adata.obsp["connectivities"]
        if not isinstance(connectivity, np.ndarray):
            # Assume scipy sparse
            connectivity = connectivity.toarray()
        binary_connectivity = (connectivity > 0).astype(float)

        # All variables now numpy arrays
        transition_matrix = self.calculate_transition_probs(
            potential=potential.squeeze(-1).cpu().numpy(),
            connectivity_matrix=binary_connectivity,
            beta=1.0
        )

        adata.obsp['transitions'] = transition_matrix
        return adata
    
    def _random_walks(self, n_trajectories):
        """
        Generate random walks in parallel on GPU.
        
        Parameters:
            T (torch.Tensor): Dense transition probability matrix of shape (n_cells, n_cells) on GPU.
            n_steps (int): Number of steps per walk.
            n_trajectories (int): Number of random walks to generate.
        
        Returns:
            torch.Tensor: Tensor of shape (n_trajectories, n_steps+1) containing the cell indices visited.
        """
        n_cells = self.P.shape[0]
        # Initialize all walks with a random starting cell for each trajectory
        current_states = torch.randint(0, n_cells, (n_trajectories,), device=self.device, dtype=torch.long)
        walks = torch.empty((n_trajectories, self.path_len), dtype=torch.int, device=self.device)
        walks[:, 0] = current_states
        
        # Simulate walks in parallel (vectorized across trajectories)
        for step in range(1, self.path_len):
            # For each trajectory, sample the next cell using its row probabilities.
            # P[current_states] gives a tensor of shape (n_trajectories, n_cells)
            next_states = torch.multinomial(self.P[current_states.to(self.device)], num_samples=1).squeeze(1)
            walks[:, step] = next_states
            current_states = next_states
            
        return walks
    def prepare_walks(self, n_trajectories: Optional[int] = None):
        """
        Prepares multiple random walks using GPU acceleration.
        
        Parameters:
            n_steps (int): Number of steps per walk.
            n_trajectories (int): Number of random walks to generate.
            connectivity (scipy.sparse matrix): Sparse connectivity matrix.
        
        Returns:
            torch.Tensor: Tensor of shape (n_trajectories, n_steps+1) with random walk trajectories on GPU.
        """
        if n_trajectories is None:
            if self.walk_config is None:
                raise ValueError("Number of trajectories must be provided when no walk config is set.")
            n_trajectories = self.walk_config.num_walks

        # Convert connectivity matrix to a dense transition matrix on GPU
        # Generate the random walks in parallel on GPU
        self.P = self.P.to(self.device)
        walks = self._random_walks(n_trajectories)
        self.P = self.P.cpu()
        self.walks = walks.cpu()



    def set_prior_transition(self, prior_time_key=None, prior_transition=None, random_state = 42):
        """
        Sets the prior cell-cell transition matrix for the model.

        Args:
            prior_time_key (str, optional): Name of the pseudotime key in adata.obs.
            phylogeny (dict, optional): Phylogeny as a parent: [child1, child2, ...] dict.
            prior_transition (np.ndarray or scipy.sparse matrix, optional): 
                Precomputed prior transition matrix. Will be stored as a torch tensor.
        """
        n_cells = len(self.adata)

        def _get_connectivity_matrix():
            """Helper: return binary connectivity matrix as np.ndarray."""
            if "connectivities" not in self.adata.obsp:
                raise KeyError(
                    "'connectivities' matrix not found in adata.obsp. "
                    "Run neighbors graph computation (e.g. sc.pp.neighbors)."
                )
            mat = self.adata.obsp["connectivities"]
            if not isinstance(mat, np.ndarray):
                mat = mat.toarray()
            return (mat > 0).astype(float)

        if prior_transition is not None:
            # Accepts dense numpy or sparse
            if not isinstance(prior_transition, np.ndarray):
                prior_transition = prior_transition.toarray()
            if prior_transition.shape != (n_cells, n_cells):
                raise ValueError(
                    f"Shape mismatch: prior_transition has shape {prior_transition.shape}, "
                    f"but expected ({n_cells}, {n_cells}) from adata. "
                    "Check that your input matrix and AnnData object refer to the same cells in the same order."
                )
            self.P = torch.from_numpy(prior_transition).float()
            print("[LSD] Prior transition matrix set from user input.")
            return

        # If prior_transition is None, try to infer from phylogeny or pseudotime
        if self.phylogeny is not None:
            A = self._create_phylogeny_matrix()
            if not isinstance(A, np.ndarray):
                A = A.toarray()
            connectivity = _get_connectivity_matrix()
            A *= connectivity
            
            self.adata.obsp["phylogeny_matrix"] = A
            row_sums = A.sum(axis=1)
            valid_cells = row_sums > 0
            if len(self.adata[~valid_cells]) !=0:
                print(f"[LSD] Removing {np.sum(~valid_cells)} cells with no transitions:")
            self.adata = self.adata[valid_cells]
            if prior_time_key is not None:
                # Use pseudotime with phylogeny
                P = self._get_transition_from_pseudotime(prior_time_key, self.adata.obsp["phylogeny_matrix"].toarray())
                print("[LSD] Prior transition matrix set from phylogeny and pseudotime.")
            else:
                raise KeyError(
                    "Run the function get_prior_transition first"
                    )
            self.P = torch.from_numpy(P).float()
            return

        if prior_time_key is not None:
            connectivity = _get_connectivity_matrix()
            P = self._get_transition_from_pseudotime(prior_time_key, connectivity)
            self.P = torch.from_numpy(P).float()
            print("[LSD] Prior transition matrix set from pseudotime and connectivities.")
            return

        raise KeyError(
            "LSD requires either a prior pseudotime, a phylogeny, or a prior transition matrix for initialization."
        )
    
    
    def _get_transition_from_pseudotime(self, time_key,connectivity):
        potential = -self.adata.obs[time_key].values
        P = self.calculate_transition_probs(potential, connectivity, beta = 50)
        return P

    def _get_all_descendants(self,cluster, descendants=None):
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
        direct_children = self.phylogeny.get(cluster, [])
        descendants.update(direct_children)
        
        # Recursively add children of children
        # for child in direct_children:
        #     descendants.update(get_all_descendants(child, phylogeny))
        
        return descendants
    
    def set_phylogeny(self, phylogeny, cluster_key):
        self.phylogeny = phylogeny
        self.cluster_key = cluster_key
    
    def _create_phylogeny_matrix(self):
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
        clusters = self.adata.obs[self.cluster_key].unique().tolist()  # Assuming cluster info is in adata.obs['cluster']
        
        # Create a dictionary mapping cells to their cluster
        cell_to_cluster = dict(zip(self.adata.obs_names, self.adata.obs[self.cluster_key]))
        
        # Precompute all descendants for each cluster to avoid repeated computation
        all_descendants = {}
        for cluster in clusters:
            all_descendants[cluster] = self._get_all_descendants(cluster)
        
        # Initialize a matrix of zeros with shape (n_cells, n_cells)
        n_cells = self.adata.shape[0]
        phylo_matrix = np.zeros((n_cells, n_cells))
        
        # Fill the matrix based on the rules
        for i, cell_i in enumerate(self.adata.obs_names):
            cluster_i = cell_to_cluster[cell_i]
            
            for j, cell_j in enumerate(self.adata.obs_names):
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


    def stream_lines(
        self,
        embedding,                          # e.g., "X_umap", "X_tsne"
        save=False,
        file_name=None,
        *,
        color="clusters",                   # obs key to color by (categorical or continuous)
        cmap=None,                          # for continuous variables (e.g., "viridis", "plasma")
        palette=None,                       # dict or list for categoricals; if None, try adata.uns[f"{color}_colors"]
        size=6.0,
        alpha=0.9,
        title=None,
        legend_loc="right",                 # "right", "on data", None (hide)
        frameon=False,
        bg_color="white",
        vmin=None,
        vmax=None,
        colorbar=True,
        show=True,
    ):
        """
        Plot CellRank streamlines in `embedding` and color points/streamlines by an obs column.

        Parameters
        ----------
        embedding : str
            E.g. "X_umap", "X_tsne", "X_diff_state" (must be present in adata.obsm)
        color : str
            Column in adata.obs to color by (categorical or continuous).
        palette : dict or list or None
            For categoricals. If None, tries adata.uns[f"{color}_colors"].
        cmap : str or None
            For continuous variables.
        vmin, vmax : float or None
            Clamp continuous colormap range.
        legend_loc : str or None
            Legend placement for categorical coloring. Use None to hide.
        colorbar : bool
            Show colorbar for continuous variables.
        size, alpha : float
            Marker size and opacity.
        frameon, bg_color, title : misc.
        save, file_name : bool, str
            If save=True, writes SVG to `file_name`. Otherwise shows the figure if `show=True`.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from cellrank.kernels import ConnectivityKernel

        adata = self.get_adata()

        # Build kernel with your precomputed transitions
        ck = ConnectivityKernel(adata)
        ck.transition_matrix = adata.obsp["transitions"]

        # Determine if `color` is categorical or continuous
        if color not in adata.obs.columns:
            raise KeyError(f"`color`='{color}' not found in adata.obs.")
        dtype = adata.obs[color].dtype
        is_categorical = (str(dtype) == "category") or (getattr(dtype, "name", "") == "category")

        # Derive palette if needed (for categorical)
        if is_categorical and palette is None:
            uns_key = f"{color}_colors"
            if uns_key in adata.uns:
                cats = list(adata.obs[color].astype("category").cat.categories)
                cols = list(adata.uns[uns_key])
                if len(cols) == len(cats):
                    palette = dict(zip(cats, cols))



        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor(bg_color)

        # Common kwargs passed to CellRank's plotter (mirrors scanpy API)
        common_kwargs = dict(
            basis=embedding,
            recompute=True,
            ax=ax,
            show=False,              # we handle showing/saving ourselves
            frameon=frameon,
            size=size,
            alpha=alpha,
            title=title,
        )

        # Coloring options
        if is_categorical:
            # Categorical: pass palette and legend settings
            plot_ax = ck.plot_projection(
                color=color,
                palette=palette,
                legend_loc=legend_loc,
                **common_kwargs
            )
        else:
            # Continuous: pass cmap, vmin/vmax, and colorbar toggle
            plot_ax = ck.plot_projection(
                color=color,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
                legend_loc=None,  # no categorical legend
                **common_kwargs
            )

        # Ensure we have an axis
        ax = plot_ax if plot_ax is not None else ax

        # Rasterize heavy artists (streamlines, scatter, etc.)
        for coll in ax.collections:
            try:
                coll.set_rasterized(True)
            except Exception:
                pass
        for ln in ax.lines:
            try:
                ln.set_rasterized(True)
            except Exception:
                pass

        # Save or show
        if save and file_name is not None:
            fig.savefig(file_name, format="svg", bbox_inches="tight")
            plt.close(fig)
        else:
            if show:
                plt.show()
            else:
                plt.close(fig)

        return ax


    def ode_solve(self, z0, time_range, num_points = None):
        t = torch.linspace(0, time_range, num_points).to(self.device)
        z = odeint(self.lsd.gradnet, z0, t)
        return z.detach()

    def propagate(self,x , time_range, num_points):
        x0 = x.to(self.device)
        z0 , _= self.lsd.x_encoder(x0)
        
        if num_points is None:
            num_points = 2*time_range
        z = self.ode_solve(z0, time_range, num_points = num_points)
        return z
    def project_z(self, z, adata):
        latent_rep = torch.tensor(adata.obsm['X_cell_state'], dtype=torch.float32).to(self.device)
        X = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(self.device)
        # Compute pairwise distances and find the nearest neighbor for each sample.
        nn = torch.argmin(torch.cdist(z, latent_rep), dim=-1).to(self.device)
        x_proj = X[nn, :]
        z_end = latent_rep[nn, :]
        return z_end, x_proj, nn
    
    def get_cell_fates(self, adata, time_range = 50, dt = 0.5, cluster_key = None, batch_size = 512, return_paths = False):
        adata = adata.copy()
        IC = torch.from_numpy(adata.X.toarray()).to(self.device)
        z_sol_batches = []
        nn_list = []
        paths = []
        for batch in tqdm(torch.split(IC, batch_size), desc="Batch Propagation", leave = False):
            z_sol = self.propagate(batch, time_range=time_range, num_points=int(time_range/dt))
            z_final = z_sol[-1, :, :]
            _, __, nn = self.project_z(z_final.to(self.device), adata)
            nn_list.append(nn)
            z_sol_batches.append(z_sol)
            if return_paths:
                _, __, path = self.project_z(z_sol.to(self.device), adata)
                paths.append(path)
        z_sol = torch.cat(z_sol_batches, dim=1)
        nn = torch.cat(nn_list, dim=0)
        if return_paths:
            paths = torch.cat(paths, dim = 1)
        adata.obs["fate"] = "Other"
        predicted_fates = adata.obs[cluster_key][nn.cpu().numpy()].values
        adata.obs["fate"] = predicted_fates
        if return_paths:   
            self.paths = paths
            self.z_sol = z_sol.detach()
            self.fates = nn
        else:
            self.z_sol = z_sol.detach()
            self.fates = nn
        return adata
    
    def _perturb(self,adata, x, gene_name, pertubation_level = 0, dt_pert = 0.1, t_unpert = 10, max_perturbations=10):
        gene_idx = adata.var_names.get_loc(gene_name)
        X = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(self.device)
        x0_pert = x.clone()
        x0_pert[:, gene_idx] = pertubation_level
        x0_unpert = x.clone()
        
        n_samples = x.shape[0]
        prev_nn_perturb = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        prev_nn_unperturb = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        steps_pert = 0
        for i in range(max_perturbations):
            steps_pert += 1
            z_perturb = self.propagate(x0_pert, time_range=dt_pert, num_points= 10) # shape: (num_points, n_samples, latent_dim)
            z_perturb_final = z_perturb[-1, :, :]          # take final latent positions
            
            # Project z_final onto the data manifold.
            # Project_z returns (projected_z, projected_x, nn)
            _, x_proj, nn_perturb = self.project_z(z_perturb_final.to(self.device), adata)
            nn_perturb = nn_perturb.to(self.device)
            prev_nn_perturb = nn_perturb.clone()

            z_unperturb = self.propagate(x0_unpert, time_range=dt_pert, num_points= 10) # shape: (num_points, n_samples, latent_dim)
            z_unperturb_final = z_unperturb[-1, :, :]          # take final latent positions
            
            # Project z_final onto the data manifold.
            # Project_z returns (projected_z, projected_x, nn)
            _, __, nn_unperturb = self.project_z(z_unperturb_final.to(self.device), adata)
            nn_unperturb = nn_unperturb.to(self.device)
            prev_nn_unperturb = nn_unperturb.clone()
            
            # Update perturbed initial condition: use the projected data point, and reapply perturbation.
            x0_pert = X[nn_perturb, :].clone()
            x0_unpert = X[nn_unperturb, :].clone()
            x0_pert[:, gene_idx] = pertubation_level
            

        final_nn_pert = prev_nn_perturb.clone()
        final_nn_unpert = prev_nn_unperturb.clone()
        x0_pert = X[final_nn_pert, :].clone()
        x0_unpert = X[final_nn_unpert, :].clone()


        z_unperturb = self.propagate(x0_unpert, time_range=t_unpert, num_points= 10)
        z_unperturb_final = z_unperturb[-1, :, :] 
        _, __, fate_unperturb = self.project_z(z_unperturb_final.to(self.device), adata)

        z_perturb = self.propagate(x0_pert, time_range=t_unpert, num_points= 10)
        z_perturb_final = z_perturb[-1, :, :]
        _, __, fate_perturb = self.project_z(z_perturb_final.to(self.device), adata)

        return fate_perturb, fate_unperturb
    
    def perturb(self,adata, x, gene_name, cluster_key,
                 pertubation_level = 0, dt_pert = 0.2, t_unpert = 15, max_perturbations=10,
                 batch_size = 512):
        
        final_nn_pert_list = []
        final_nn_unpert_list = []

        # Divide x0 into batches.
        for batch in torch.split(x, batch_size):
            # Call your provided perturb_until_convergence on the current batch.
            fate_perturb, fate_unperturb =  self._perturb(adata = adata, x =batch, gene_name = gene_name,
                                                      pertubation_level = pertubation_level, dt_pert = dt_pert,
                                                        t_unpert = t_unpert, max_perturbations=max_perturbations)
            final_nn_pert_list.append(fate_perturb)
            final_nn_unpert_list.append(fate_unperturb)
        
        # Concatenate attractor indices from all batches.
        final_nn_pert_all = torch.cat(final_nn_pert_list, dim=0).cpu().numpy()
        final_nn_unpert_all = torch.cat(final_nn_unpert_list, dim=0).cpu().numpy()
        perturbed_fates = adata.obs[cluster_key].iloc[final_nn_pert_all].values
        unperturbed_fates = adata.obs[cluster_key].iloc[final_nn_unpert_all].values

        return perturbed_fates, unperturbed_fates
