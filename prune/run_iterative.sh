#!/usr/bin/env bash


# iterative method
python "1_iter_train_baseline.py"
python "2_iter_analyse.py"
python "3_iter_retrain_sparse_model.py"
python "4_iter_prune.py"
python "5_iter_compare.py"

