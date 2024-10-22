# Time series clustering with Temporal Convolutional Neural Network

## Description

This repository implements a model-based clustering technique to cluster samples from temporal Poisson point processes, based on their latent intensity functions. We model the intensity functions for each cluster as a function of the component point processes using a causal Temporal Convolutional Neural Network, as described in [Bai. et. al](https://doi.org/10.48550/arXiv.1803.01271). The cluster membership probabilities are likewise modeled as a function of the input point processes. A negative log-likelihood loss is used, and in particular, a Poisson point process likelihood. Learning is done using gradient descent, which is implemented in PyTorch. As such, this implementation takes full advantage of GPU acceleration and parallelization. We applied the method to cluster simulated Poisson process data, from 3 clusters defined by 3 different intensity functions, and were able to recover the clusters by fitting the model to the data.

## File Structure

## Getting Started

* ```tcn.py```: Contains the main source code for the causal, temporal convolutional neural network, and the negative log-likelihood loss function.
* ```general_functions.py```: Includes utility and helper functions for the model, such as logging and plotting functions.
* ```main_tcn.py```: Script to run the clustering algorithm on sample simulated data.
* ```main_learn_initial_outputs.py```: Contains code to learn the intensity functions for each cluster.
* ```main_learn_initial_maps.py```: Contains code to learn the cluster membership probabilities.
* ```main_finetune_maps.py```: Contains code to jointly learn the intensity functions and cluster membership probabilities.

### Prerequisites

```
      - Python 3.8 or higher
      - allensdk==2.15.1
      - cython==0.29.35
      - distro==1.8.0
      - h5py==3.8.0
      - hdmf==3.4.7
      - matplotlib==3.4.2
      - numpy==1.23.5
      - pandas==1.5.3
      - scipy==1.10.1
      - seaborn==0.12.2
      - torch==2.2.1
```

### Installation

1. Clone the repository:

```
git clone https://github.com/Tolani-O/Time_series_clustering_with_temporal_convolutions.git
cd Time_series_clustering_with_temporal_convolutions
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```
pip install -r requirements.txt
```

### Usage

Simply run ```main_tcn.py```.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any feature additions or bug fixes.

## License

This project does not have a license at this moment.

## Contact

For any questions or issues, please open an issue on this repository or contact the repository owner at [Tolani-O](https://github.com/Tolani-O).

## Version History

* 0.1
    * Initial Release
