# CogPonder

An Interoperable Multitask Model of Response Times in Cognitive Tests


## Setup

> **Note**
> This codebase is written in Python 3.10.

To install the dependencies, you need [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (or even better, [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)). Then, create a new environment with the dependencies as follows:


```bash
mamba create -f environment.yml
mamba activate ponder
```

## Notebooks

The notebooks are organized as follows:

- `Loss Functions.ipynb`:

## Data

### N-Back

The SRO-2back interface provides the following features from the *Self-Regulation Ontology* study:

- input `X_{ij}`: previous N+1 symbols for the subject $i$ and trial $j$. The last symbol is the current trial. For each subject, $X_i$ is a 2-dimensional vector of integers of shape ($N_{\text{trials}}$, 3).
- `trial_type`: Correct match, incorrect match, correct-non-match, incorrect-non-match for each trial $i$.
- `is_target`: whether the trial $i$ is a match; it is a boolean.
- output `response`: the response of the subject for the trial i; it is a boolean.
- `response_step`: the response step of the subject for the trial i; Response step is an integer and represents *response times* in 50ms steps. This step duration is a hyperparameter of the data module.

### Stroop

The SRO-stroop interface provides the following features from the *Self-Regulation Ontology* study:


- input `X_{ij}`: Color, word, and congruency for the subject $i$ and trial $j$. For each subject, $X_i$ is a 2-dimensional vector of integers of shape ($N_{\text{trials}}$, 3).
- condition
- response
- response_step

## Acknowledgements
TODO

## License
TODO
