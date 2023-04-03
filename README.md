# CogPonder

An Interoperable Multitask Model of Response Times in Cognitive Tests


## Setup

> **Note**
> This codebase is written in Python 3.10.

To install the dependencies, you need [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (or even better, [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)). Then, create a new environment with the dependencies as follows:


```bash
mamba create -f env.yml
mamba activate cogponder
dvc update --rev master -R data  # import the SRO data using DVC
```

## Notebooks

The notebooks are organized as follows:

- `notebooks/1 N-Back.ipynb`: Running the agent on the 2-back test using some manual choices of hyperparameters.
- `notebooks/2 Stroop.ipynb`: Running the agent on the Stroop test using some manual choices of hyperparameters.
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
