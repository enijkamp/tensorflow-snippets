#!/bin/bash
git clone https://github.com/tensorflow/models.git

virtualenv --system-site-packages -p python3 tf-venv3
source tf-venv3/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu==1.5.0

python models/tutorials/image/mnist/convolutional.py

# Titan X Pascal (local) -> 2.7ms
# Quadro P5000 (Paperspace) -> 4.5ms
# Quadro P6000 (Paperspace) -> 4.0ms
# Tesla V100 (Paperspace)   -> ?
