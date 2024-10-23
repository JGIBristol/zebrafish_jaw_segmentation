# zebrafish_jaw_segmentation
Segmentation of Zebrafish Jawbones from uCT scans using a U-Net

![UT/IT](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/fast_tests.yml/badge.svg?branch=main)
![Tests](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/system_tests.yml/badge.svg?branch=main)
![Pylint](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/pylint.yml/badge.svg?branch=main)
![Format](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/format.yml/badge.svg?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Project Description
This project aims to segment the jawbone of zebrafish from micro-Computed Tomography (uCT) scans using a 3D U-Net model.


Manual segmentation takes a long time - even an imperfect automated segmentation should massively speed up the process,
(e.g. it's pretty easy to manually erode bits that aren't right) and let us do some more analyses, like
statistical analyses on the morphology and finite element analysis.

It's heavily based on the work performed by Wahab Kawafi in his PhD thesis[^1].

The segmentation model is implemented in `pytorch`, using the `monai` `AttentionUnet` architecture.

## Usage
In the below examples, shell commands are in `bash` on Linux.
If you're not using linux, things might not work but also they might be fine.

Also, becuase `python` environments are a huge mess, I've elected to prepend each python command with
`PYTHONPATH=$(pwd)` which basically tells the python interpreter to look in the current directory for modules
to import.
You'll need to do this so it can import the `fishjaw` module, which is where all the code for this project lives.
This means that I run scripts by typing e.g.

```
PYTHONPATH=$(pwd) python my_cool_script.py
```
on the command line.

This isn't a perfect solution, but it's passable. Long-term, we might want to publish `fishjaw` as a package on pypi
but we're not quite there yet.

If you're on Windows and need to type something else, probably google it idk


### Environment
The code here is written in Python.

I use `conda` to manage my python environment.

```
conda env create -f environment.yml
```

You'll need an nvidia GPU to make this code work; a lot of it relies on CUDA.

Old versions of conda are painfully, unusably slow, but modern `conda` versions are fast at solving
the environment so are useful for rapidly getting stuff done.
If it is taking you a long time to create an environment or install things in your `conda` environment,
try updating or creating a fresh install of `conda` (consider using miniconda).

### Setting up the data
The first thing to do is convert the labelled data to DICOM files.

You can do this by running the `scripts/create_dicoms.py` script:

```
PYTHONPATH=$(pwd) python scripts/create_dicoms.py
```

#### More background on the data setup
This is a file format for medical imaging that keeps lots of related things together--
in our case, it's mostly useful because it lets us store our image and label
together in one file which will guarantee we won't do anything silly like accidentally
match up the wrong label to an image.

The raw data lives on the group's RDSF drive.
You can access this by asking Chrissy, and then by learning how to do that.
The labels live in the `1Felix and Rich make models/Training dataset Tiffs` directory here.
The original images live elsewhere - 3D TIFFs are in `DATABASE/uCT/Wahab_clean_dataset/TIFS/`.

There are also some 2D TIFFs in the `DATABASE/uCT/Wahab_clean_dataset/` directory somewhere, but
I would recommend dealing with DICOMs wherever possible.
If you have to deal with TIFF files try to read/write 3D TIFFs where possible instead of dealing
with lots of 2d TIFFs, because it's much faster to read one big file than many small ones.

### Training the model
- Train the model or load some weights that might be somewhere using `train_model.py`

### Running inference
- Perform inference using `inference_example.py`

### Going further
- You can turn the voxels from the inference into a mesh by learning how to do that

#### Config File
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
[^1]: Kawafi, A., 2023. Quantitative 3D Bioimaging Using Machine Learning (Doctoral dissertation, University of Bristol).t:
