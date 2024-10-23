# zebrafish_jaw_segmentation
Segmentation of Zebrafish Jawbones from uCT scans using a U-Net

![UT/IT](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/fast_tests.yml/badge.svg?branch=main)
![Tests](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/system_tests.yml/badge.svg?branch=main)
![Pylint](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/pylint.yml/badge.svg?branch=main)
![Format](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/format.yml/badge.svg?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TODO
- [ ] README (including credit for Wahab)
- [x] environment.yml
- [ ] GitHub actions - tests, lint, typing
  - [x] linting
  - [x] formatting
  - [ ] typing
  - [ ] tests
  - [ ] actual useful tests

- [x] Model train script + example notebook
- [x] config files for hard-coded config, user config
- [x] hyperparameter tuning script + example notebook
- [x] mesh script + example notebook
- [ ] write some actual tests that test functionality
- [ ] compare accuracy to a sensible baseline
- [ ] latex paper with build script
- [ ] full documentation

Scripts:
- [x] Create DICOMs
- [x] View DICOMs

## Project Description
This project aims to segment the jawbone of zebrafish from micro-Computed Tomography (uCT) scans using a 3D U-Net model.
Manual segmentation takes a long time - even an imperfect automated segmentation should massively speed up the process,
and let us do some more statistical analyses.
This might include looking at how the mechanical properties of the jawbone change between young, aged, wildtype and
mutant fish to investigate whether bone degeneration is the same in aged and mutant fish.

It's heavily based on the work performed by Wahab Kawafi in his PhD thesis[^1].

The segmentation model is implemented in `pytorch`, using the `monai` `AttentionUnet` architecture.

## Usage
I haven't made a nice API or anything for a general random person to use this yet. Sorry!

### For Users
- Train the model or load some weights that might be somewhere using `train_model.py`
- Perform inference using `inference_example.py`
- You can turn the voxels from the inference into a mesh by learning how to do that

#### Config Files
There are two config files for this project - `config.yml` and `userconf.yml`.

 - You probably don't need to edit anything in `config.yml`.
 - `userconf.yml` contains things that you might want to edit - model parameters etc.
    The training script reads the parameters from this file before it begins training.
    The model also gets information from here during inference - which means you can't change
    it between training + inference. Maybe this is bad and stupid, and I should write a model class
    that holds the model and the config used to train it?

### For Developers
i am the developer

## Footnotes
[^1] Kawafi, A., 2023. Quantitative 3D Bioimaging Using Machine Learning (Doctoral dissertation, University of Bristol).t:
