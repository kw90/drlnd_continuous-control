# Solving the DRLND Unity Reacher Environment

Train an agent to solve the `Reacher` Unity Environment from the Deep Reinforcement
Learning Nanodegree on Udacity.

- TODO: Add trained example run
- TODO: Add reward plot
- TODO: Add description and link detailed report

## Prerequisites

- `conda` or `miniconda` (recommended)
- `make`
- Download the environment that matches your OS following the *Getting Started* from the DRLND
  [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started)
  and unpack it in the root of this project

## Install Environment

### Automated Install

Simply run `make install` to install all requirements in a `conda` environment
called `drlnd_control`.

### Manual Install

Create a `conda` environment called `drlnd_control` with Python3.6 and activate it
using the following commands

```zsh
conda create --name drlnd_control python=3.6
conda activate drlnd_control
```

Then install the requirements file `requirements.txt` and install the drlnd_control
ipykernel.

```zsh
pip install -r $(PWD)/requirements.txt
python -m ipykernel install --user --name drlnd_control --display-name "drlnd_control"
```

## Run the Code

Next, run `make start` to start the Jupyter notebook server and use your favorite
browser to navigate to
[http://localhost:8888/?token=abcd](http://localhost:8888/?token=abcd).

### Train an Agent

- TODO: Docs how to train an agent

### Watch a Trained Agent

- TODO: Docs how to watch a successful agent
