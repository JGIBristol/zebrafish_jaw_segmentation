#!/bin/bash
# Shell script for training lots of models with the same configuration
# I've just kept it in the root of the repo because we need to restructure everything anyway

# Backup the original config file
cp userconf.yml userconf.yml.backup

for i in {0..19}; do
    echo "Training model attempt_${i}.pkl (iteration $((i+1)) of 6)"
    
    # Update the model_path in the YAML file
    sed -i "s/^model_path: .*/model_path: \"attempt_n${i}.pkl\"/" userconf.yml
    
    # Run the training script
    PYTHONPATH=$(pwd) python scripts/train_model.py > "logs/attempt_${i}.log" 2>&1
    
    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "Training failed for attempt_${i}.pkl"
        # Restore original config and exit
        cp userconf.yml.backup userconf.yml
        exit 1
    fi
    
    echo "Completed training for attempt_${i}.pkl"
    echo "----------------------------------------"
done

# Restore the original config file
cp userconf.yml.backup userconf.yml
rm userconf.yml.backup
