#!/usr/bin/env bash


# iterative method
python "1_train_baseline.py"
python "2_analyse.py"
python "3_retrain_sparse_model.py"
python "4_prune.py"
python "5_compare.py"

