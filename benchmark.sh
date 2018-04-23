#!/bin/bash
git clone https://github.com/tensorflow/models.git

virtualenv --system-site-packages -p python3 tf-venv3
source tf-venv3/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu==1.5.0

python models/tutorials/image/mnist/convolutional.py

# QUADRO P5000 -> 5ms