# CogPonder

> [!WARNING]
> This is a work-in-progress. Model architecture and results are subject to change as we continue to develop and refine the model.

CogPonder is a flexible, differentiable model of cognitive control that is inspired by the Test-Operate-Test-Exit (TOTE) architecture in psychology and the PonderNet framework in deep learning. CogPonder functionally decouples the act of control from the controlled processes by introducing a controller that wraps around any end-to-end deep learning model and decides when to terminate processing and output a response, thus producing both a response and response time.


[CogPonder: Towards a Computational Framework of General Cognitive Control
](https://2023.ccneuro.org/proceedings/0001148.pdf)


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

