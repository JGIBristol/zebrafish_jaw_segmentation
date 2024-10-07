#!/bin/bash
set -ex

CONDA_DIR="${HOME}/fishconda"

# Check if the conda dir in the home directory exists
if [ -d ${CONDA_DIR} ]; then
    echo "The conda directory already exists - delete it if you want to install a fresh copy"
    exit 1
fi

# Check if the miniforge3.sh file exists
if [ ! -f miniforge3.sh ]; then
    wget -O miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/miniforge3-$(uname)-$(uname -m).sh"
else
    echo "miniforge3.sh already exists - delete it if you want to download a fresh copy"
    exit 1
fi

# Run the setup script
bash miniforge3.sh -b -p ${CONDA_DIR}

source "${CONDA_DIR}/etc/profile.d/conda.sh"
source "${CONDA_DIR}/etc/profile.d/mamba.sh"

conda activate

# Install the environment
conda env create -f environment.yml
conda activate zebrafish_jaw_segmentation
which python