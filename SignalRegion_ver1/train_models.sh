#!/bin/sh

python train_model.py -c 1E2Mu -m MHc70_MA15
python train_model.py -c 1E2Mu -m MHc70_MA40
python train_model.py -c 1E2Mu -m MHc70_MA65

python train_model.py -c 1E2Mu -m MHc100_MA15
python train_model.py -c 1E2Mu -m MHc100_MA25
python train_model.py -c 1E2Mu -m MHc100_MA60
python train_model.py -c 1E2Mu -m MHc100_MA95

python train_model.py -c 1E2Mu -m MHc130_MA15
python train_model.py -c 1E2Mu -m MHc130_MA45
python train_model.py -c 1E2Mu -m MHc130_MA55
python train_model.py -c 1E2Mu -m MHc130_MA90
python train_model.py -c 1E2Mu -m MHc130_MA125

python train_model.py -c 1E2Mu -m MHc160_MA15
python train_model.py -c 1E2Mu -m MHc160_MA45
python train_model.py -c 1E2Mu -m MHc160_MA75
python train_model.py -c 1E2Mu -m MHc160_MA85
python train_model.py -c 1E2Mu -m MHc160_MA120
python train_model.py -c 1E2Mu -m MHc160_MA155

