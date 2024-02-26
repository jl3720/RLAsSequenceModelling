# UDRL Deep Learning Project

This repo hosts the code used to train the agents and generate the figures for the "Upside-down Reinforcement Learning" project. The aim of the project was to explore the dynamics of self-play to alleviate the issue of exploration when casting RL as a supervised-learning project. 

You can checkout the report [here](https://github.com/jl3720/RLAsSequenceModelling/blob/master/UDRL_Submission.pdf). 

We were awarded a **5.5 out 6.0** for the report.

## Overview
- **connect4/**
    - Contains **connect4.py**
    - `Connect4` is a `gym` like environment interface to control agent-environment interactions.
    - `GameManager` controls the logic allowing the playing of multiple games in parallel.
    - `Player` is a base class that baseline and agent classes inherit from. Implements `steps()` to return actions.

- **deeplearning/**
    - `ReplayBuffer` stores agent experiences and allows sampling to create batches of training data.
    - `BF` is the Behaviour Function, i.e. MLP model and inherits from `Player` and `torch.nn.Module`. You can use `steps()` to get action probabilities given the state and desired reward. Also implements saving and loading.
    - `League` controls the tournament logic between the pool of agents, allowing for self-play and training.

- **picture_maker.py**
    - Allows visualisation of action probabilities over the current board state.

- **data/**
    - Example of default logging of buffer, model and ELO histories.
    - Allows visualisations in **visualise.ipynb**.

## Install
- Use `pipenv` to create a virtual environment and install packages from the `pipfile`.
- We use Python 3.10.
- Ensure you are in the project root.
```bash
$ cd <PROJECT_ROOT>
$ ls
>>> connect4    data    deeplearning    README      etc...
$ pip install pipenv
$ pipenv install
```
Or if that fails use `conda`:
```bash
$ conda create -n "udrl" python=3.10 pip
$ conda activate udrl
$ pip install numpy torch pympler torchvision torchaudio matplotlib ipykernel
```

## Scripts and notebooks
- **train.py**: Script used to run training seasons. Logging and checkpointing are handled by the `League` class.<br>
    `python train.py --help` to view options.
    - Args: 
    - --buffer-path: Path to load buffer from.
    - --season: Season to start with.
    - --elo-log-dir: directory to log elo history
    - --num-seasons: number of seasons to train for
- **train.ipynb**: Notebook demonstrating how to train agents, load logs and visualise ELO evolution as in report.
- **visualise.ipynb**: Notebook demonstrating how to load a trained agent, rollout actions and visualise predicted action probability distributions as in report.
