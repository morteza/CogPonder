# CogPonder

An Interoperable Multitask Model of Response Times in Cognitive Tests


## Setup

To install the dependencies, you need [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (or even better, [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)). Then, create a new environment with the dependencies as follows:


```bash
mamba env create -f environment.yml
mamba activate cogponder
dvc update --rev master -R data  # import the SRO data using DVC
```

To install additional GPU dependencies run the following:

```bash
mamba env update -f environment_gpu.yml --prune
```

## Notebooks

The notebooks are organized as follows:

- `notebooks/N-Back.ipynb`: Running a single-task agent on the 2-back test using arbitrary hyperparameters.
- `notebooks/Stroop.ipynb`: Running a single-task agent on the Stroop test using arbitrary hyperparameters.

## Data

### N-Back

See the [adaptive N-back task data description](data/Self_Regulation_Ontology/README.md#adaptive-n-back).


### Stroop

See the [Stroop task data description](data/Self_Regulation_Ontology/README.md#stroop-task).

## Acknowledgements
TODO

## License
TODO
