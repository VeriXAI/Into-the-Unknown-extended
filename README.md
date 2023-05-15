# Into the Unknown (Extended)

This repository contains the implementation and data used in the paper "Into the Unknown: Active Monitoring of Neural Networks (Extended)".

# Installation

We use Python 3.6 but other Python versions may work as well.
The package requirements that need to be installed are found in the file `requirements.txt`.

Since the datasets are large and have mostly been used in our previous work, we do not include most of them here.
You need to manually download them (see the links below) and extract them to the `data` folder of this repository.

Modify the file called `paths.txt` in the base folder, which contains two lines that are the paths to the model and dataset folders:

```
.../models/
.../data/
```

Here replace the `...` with the absolute path to your clone of the repository.

## Links to dataset files

- [`MNIST`](https://github.com/VeriXAI/Outside-the-Box/tree/master/data/MNIST)
- [`Fashion MNIST`](https://github.com/VeriXAI/Outside-the-Box/tree/master/data/Fashion_MNIST)
- [`GTSRB`](https://github.com/VeriXAI/Outside-the-Box/tree/master/data/GTSRB) (You need to manually extract the file `train.zip` because the content is too large for GitHub.)


# Recreation of the results

To obtain the results from the conference version of the paper [Into the Unknown: Active Monitoring of Neural Networks](https://arxiv.org/pdf/2009.06429), published at [RV 2021](https://uva-mcps-lab.github.io/RV21/), see [this repository](https://github.com/VeriXAI/Into-the-Unknown).

Below we describe how to obtain the results shown in section 7.3 of the journal version of the paper.
Moreover, all models, data and figures can be found in `abstraction_trainer_experiment_results`.


## Reproduce the Experiment

To generate the models and the data used in the experiments, run `run/train_experiment_into_the_unknown_extended.py`.


## Evaluation

To reproduce the figures found in section 7.3 of the paper, run `run/run_experiment_into_the_unknown_extended.py`.

