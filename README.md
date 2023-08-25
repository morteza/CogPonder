# CogPonder

> [!WARNING]
> This is a work-in-progress. Model architecture and results are subject to change as we continue to develop and refine the model.

An interoperable multitask model of response times in cognitive tests


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

- `notebooks/N-Back.ipynb`: training a single-task 2-back agent.
- `notebooks/Stroop.ipynb`: training a single-task Stroop agent.

## Data

see [data/Self_Regulation_Ontology/README.md](data/Self_Regulation_Ontology/README.md)

