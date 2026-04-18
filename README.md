# talking-heads | Graph Attention Neural Operators (GANO)

**This repository is a work in progress**. The goal is to use Graph Neural Networks for processing sensor data (for data assimilation) and then projecting to an output grid using attention.

"Talking": Graph neural network message passing
"Heads": multi-head attention
= "Talking Heads".

## Neural Operators

Neural Operators map between infinite-dimension function spaces rather than vectors (like traditional neural networks). In this package, this is specifically applied to the data-assimilation domain. I.e. mapping from sparse observations to a full gridded field - one function space to another. 

## Graph Neural Networks

Graph Neural Networks allow deep learning over graph-structured data. This has been applied to many domains - including weather forecasting in Graph-Cast. Graph-Cast places graph nodes on a fixed lattice structure around the globe, and uses graph-convolutions and other neural operations to produce a forecast across the lattice structure (nodes).

## Graph Attention Neural Operators

To apply Graph Neural Networks to the data assimilation domain, we attempt to model sparse observations in a graph structure. These can be sensor readings, forecasts at gridded points etc. Graph Neural Network operations can be applied over this lattice to produce a latent representation of the data to assimilate. This architecture's point of difference is how the graph is projected into a gridded output, using attention. This is done by attending over the graph nodes for each point on the output grid.

## Future Updates (work in progress)

- Implementation of Graph Convolutions
- Options for handling temporal data in the GNN in two situations:
    - Each node has a single observation, but recorded at different times.
    - Each node recieves multiple observations over time.
- Attention across the graph structure, and other static gridded fields (e.g. topography)
- Handling moving sensors
