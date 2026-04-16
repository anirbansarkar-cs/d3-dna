a user of the d3_dna package should be able to:

- look at the minimal example and implement their datalaoder, train their own diffusion model (and choose thei diffusion hyperparameters in the config.yaml) on their own dataset
    - to this point, it's not clear what the following does in the config.yaml given that the dataloading is implemented separately in the data.py file:
dataset:
  name: lentimpra
  sequence_length: 230
  num_classes: 4
  signal_dim: 1  # Single regression target (regulatory activity)

- send the model to train across multiple gpus in the train.py script. this already works in ~/D3_DNA_Discrete_Diffusion but it's not set up to work here with the following configs in the config.yaml file for each experiment:ngpus: 2
nnodes: 1
tokens: 4  # DNA alphabet: A, C, G, T

- get samples from a trained model in a sample.py file

- get the 4 basic evaluations of the functional and compositional metrics in an evaluate.py script 

the current setup of the config.yaml and how these parameters interact with modules importer from d3_dna is very unclear.you need to think about the ideal workflow first, and propose a refactor plan from this.
