# CogPonder

An Interoperable Multitask Model of Response Times in Cognitive Tests


## Setup

> **Note**
> This codebase is written in Python 3.10.

To install the dependencies, you need [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (or even better, [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)). Then, create a new environment with the dependencies as follows:


```bash
mamba create -f environment.yml
mamba activate ponder
dvc update -R data  # download the data using DVC
```

## Notebooks

The notebooks are organized as follows:

- `notebooks/Fixed Hyperparameters.ipynb`: Running the pipeline with some common choices of hyperparameters.
- `notebooks/Hyperparameter Tuning.ipynb`: Uses Ray Tune to optimize the hyperparameters.

## Data

### N-Back

See the [adaptive N-back task data description](data/Self_Regulation_Ontology/README.md#adaptive-n-back).


### Stroop

See the [Stroop task data description](data/Self_Regulation_Ontology/README.md#stroop-task).

## Acknowledgements
TODO

## License
TODO
