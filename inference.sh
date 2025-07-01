#!/bin/bash
# Shell script for performing inference with lots of the models
# I've just kept it in the root of the repo because we need to restructure everything anyway

set -e

for i in {0..19}; do
    # Run the training script
    PYTHONPATH=$(pwd) python scripts/compare_segmentations.py attempt_n${i}.pkl > "logs/attempt_${i}_inference.log" 2>&1
done
