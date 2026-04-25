from talking_heads.models.base import GraphAttentionNeuralOperator
from typing import Literal, List

# Utilities for creating GANO models with different configurations.
# There is no reason you can't create a GANO from the base class, but these functions 
# provide a convenient interface.

def create_gano(
        data_in_dim: int,
        positional_dim: int,
        data_out_dim: int,
        latent_dim: int = 64,
        attn_radius: float = None,
        architecture: str = 'meanvar-kgnn',
        distance_encoding: List[Literal['q_pos', 'o_pos', 'rel', 'rbf', 'fourier']] = ['q_pos', 'o_pos', 'rel'],
        gnn_layers: int = 2,
        gnn_k: int = 4,
        gnn_r: float = 1.0,
        gnn_self_loops: bool = True,
        activations: dict = None # activations dictionary with keys 'encoder', 'gnn', 'kernel', 'decoder' and values as activation function names (e.g. 'ReLU', 'Tanh'). These are parsed directly as pytorch modules, so ensure the function names match a pytorch activation function (case sensitive). If a key is missing, it will default to 'ReLU' for that component.
    ):
        '''
        A utility function to create GANO models with different configurations. There is no reason the model can't be created 
        directly from the GraphAttentionNeuralOperator class, but this provides a convenient interface.

        Args:
            data_in_dim: Dimension of input data at each observation point (e.g. for a node with features: temperature, wind_u, wind_v, humidity, then data_in_dim = 4).
            positional_dim: Dimension of positional information (e.g. 2 for interpolation in 2D space. I see no reason this should be larger than 3).
            data_out_dim: Dimension of output data at each query point. ()
            latent_dim: Default: 64. Dimension of latent node features in the GNN. 
            attn_radius: Default: None. Attention radius for local attention. If None, global attention is used.
            architecture: Default: 'meanvar-kgnn'. A string specifying the model architecture. This is a string in the format: outputmode-?gnnarch. The order of the components in the string does not matter.
                - outputmode: 'meanvar' or 'mean' (specifies whether the model outputs both mean and variance for uncertainty estimation, or just the mean)
                - gnnarch (optional): 'gnn', 'kgnn', 'rgnn'. 'gnn' is a GNN with KNN graph construction, 'kgnn' is a GNN with KNN graph construction, and 'rgnn' is a GNN with radius graph construction. If gnnarch is not specified, no GNN is used and the kernel operates directly on the encoded node features. If gnnarch = 'kgnn', then the gnn_r parameter is not used. Vice versa if gnnarch = 'rgnn'.
                An example architecture string is 'meanvar-kgnn' which specifies a model that outputs mean and variance and uses a GNN with KNN graph construction. Another example is 'mean-rgnn' which specifies a model that outputs just the mean and uses a GNN with radius graph construction. If the architecture string is just 'meanvar', then the model outputs mean and variance but does not use a GNN (the kernel operates directly on the encoded node features).
            distance_encoding: Default: ['q_pos', 'o_pos', 'rel']. A list specifying which types of distance encoding to use in the kernel. Options include 'q_pos' (query position), 'o_pos' (observation position), 'rel' (relative position), 'rbf' (radial basis function encoding), and 'fourier' (fourier feature encoding).
            gnn_layers: Default: 2. Number of layers in the GNN (if used).
            gnn_k: Default: 4. Number of neighbors to connect in the GNN (if used).
            gnn_r: Default: 1.0. Radius for radius graph construction (if used).
            gnn_self_loops: Default: True. Whether to include self-loops in the GNN graph construction (if used).
            activations: Default: None. A dictionary specifying activation functions for different components of the model. Keys should be 'encoder', 'gnn', 'kernel', 'decoder', and values should be strings corresponding to PyTorch activation functions (e.g. 'ReLU', 'Tanh'). If a key is missing, it will default to 'ReLU' for that component.
        '''
        
        # ---- Model Architecture Parsing ----

        arch = architecture.lower().split('-')
        if 'meanvar' in arch:
            output_mode = 'MeanVar'
        elif 'mean' in arch:
            output_mode = 'Mean'
        else:
             print("Model output mode not specified in architecture string. Defaulting to 'Mean'.")
             output_mode = 'Mean'

        use_gnn = ('gnn' in arch) or ('kgnn' in arch) or ('rgnn' in arch)
        gnn_arch = 'k' if (('gnn' in arch) or ('kgnn' in arch)) else 'r' if ('rgnn' in arch) else None

        # ==== Activation Functions ====

        # create an activations dictionary with default value 'ReLU' for all components if not provided
        if activations is None:
            activations = {
                'encoder': 'ReLU',
                'gnn': 'ReLU',
                'kernel': 'ReLU',
                'decoder': 'ReLU'
            }
        else:
            # fill in any missing keys with default value 'ReLU'
            for key in ['encoder', 'gnn', 'kernel', 'decoder']:
                if key not in activations:
                    activations[key] = 'ReLU'

        model = GraphAttentionNeuralOperator(
            in_dim_obs=data_in_dim,
            pos_dim=positional_dim,
            latent_dim=latent_dim,
            out_dim=data_out_dim,
            radius=attn_radius,
            output_mode=output_mode,
            distance_encoding=distance_encoding,
            gnn_arch=gnn_arch,
            gnn_layers=gnn_layers,
            gnn_k=gnn_k,
            gnn_r=gnn_r,
            gnn_self_loops=gnn_self_loops,
            activations=activations
        )

        return model

