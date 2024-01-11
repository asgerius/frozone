# Float-zone Process Modelling Using Machine Learning

This repository contains the code developed for my [master's thesis](https://asgerius.github.io/frozone-paper/main.pdf).

It was developed for Python 3.11 running on Linux, but it might well work on other platforms.
The following script reproduces the experiments in the report.

```sh
# Install dependencies
pip install -r requirements.txt

# Paths for where data and models should go
export DATA=<data path here>
export EXPERIMENTS=<path to put trained models and experiments at>

# Generate data
python frozone/data/generate_simulated_data.py $DATA

# Train models
python frozone/train/run.py $EXPERIMENTS -c configs/train.ini --data-path $DATA

# Evaluate a trained model using different controller settings
python frozone/eval/simulated_control.py $EXPERIMENTS/GTN_S -c configs/eval.ini
```
