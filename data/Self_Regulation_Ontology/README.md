
# "Self-Regulation Ontology" Dataset

## Stroop Task

> On each trial, subjects will be presented with a color word written in ink that is either congruent with the word, or incongruent with the word (e.g., blue written in blue =congruent, blue written in red = incongruent). Subjects are instructed to quickly and accurately respond via key press what the ink color of the word is.
https://expfactory.org/experiments/stroop/


For each trial, we use the following features:

### TODO: the following table needs to be updated (See StroopSRO data module)

SRO Feature | Description | CogPonder Name
---|---|---|
`worker_id` | the identifier of the participant. | `subject_ids`
`...` | the trial number. | `trial_ids`
`stim_color` | the ink color of the word (blue, green, red). | `stimuli[...,1]`
`stim_word` | the word (blue, green, red). This is the expected response. | `stimuli[...,1]`
`condition` | congruent or incongruent. | `trial_types`
`correct` | whether the response was incorrect=0 or correct=1. | `correct_responses`
`key_press` | response, which is expected to be the color of the word (blue=66, green=71, red=82). | `responses`
`rt` | response time in millis. | `response_steps`



## Adaptive N-Back

> A variant of the n-back task, each trial consists of letters presented in successive order. Subjects must press a button if the current letter matches the letter that occurred n trials ago (regardless of capitalization). N varies according to a staircase tracking method that increases as subjects accurately respond and decreases as subjects make errors. https://expfactory.org/experiments/adaptive_n_back/

### TODO: the following table needs to be updated (see NBackSRO data module)

The SRO-2back interface provides the following features from the *Self-Regulation Ontology* study:

- input `X_{ij}`: previous N+1 symbols for the subject $i$ and trial $j$. The last symbol is the current trial. For each subject, $X_i$ is a 2-dimensional vector of integers of shape ($N_{\text{trials}}$, 3).
- `trial_type`: Correct match, incorrect match, correct-non-match, incorrect-non-match for each trial $i$.
- `is_target`: whether the trial $i$ is a match; it is a boolean.
- output `response`: the response of the subject for the trial i; it is a boolean.
- `response_step`: the response step of the subject for the trial i; Response step is an integer and represents *response times* in 50ms steps. This step duration is a hyperparameter of the data module.

# License

> From the original repository: https://github.com/poldrack/Self_Regulation_Ontology/

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


This folder contains all of the data releases and derived data. 
Each folder is named with the sample name and the release date. The samples are:
* Discovery: 200 subjects assigned to discovery set
* Validation: 300 validation subjects
* Full: Full sample of all complete subjects
* Failed: non-completed subjects

These data should not be changed once being committed.  Any additional derived data should be placed in the appropriate
directory underneath the Derived_data folder.