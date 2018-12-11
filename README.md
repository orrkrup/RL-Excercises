# DRL Course - Wet Excercise
This repository contains work and results for the midterm exercise for the DRL Course, Winter 18-19 semester.

## Contents
Three packages each hold a `run.py` file for one of the trained models:
* DQN solving Taxi-v2
* Policy Gradient (A2C) solution of Taxi-v2
* DQN visual solution of Acrobot-v1

Additionally, a PDF report is included as drl_mid.pdf. 

## Usage
Clone the repository to your local environment. To install, type `pip install .` in the main directory. 
To view results and evaluate model, from the main directory run `./package_name/run.py`.
Each package also contains a training file, which can be used to train a new model.

## Requirements
* PyTorch 0.4
* gym

We ran our code on Python 3.6. It will probably work on other python versions as well, although it was not tested for any of them.

