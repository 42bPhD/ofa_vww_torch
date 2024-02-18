#!/usr/bin/env bash

# one-step method
python "0_setup_env.py"
python "1_train_baseline.py"
python "2_search.py"
python "3_retrain_slim_model.py"
