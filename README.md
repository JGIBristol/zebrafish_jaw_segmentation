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
This data lives on the Zebrafish Osteoarthritis group's RDSF drive, so you'll need access to this first.

You can convert the data to DICOMs by running the `scripts/create_dicoms.py` script:
```
PYTHONPATH=$(pwd) python scripts/create_dicoms.py
```
This requires you to have specified the location of your RDSF mount in `userconf.yml`--
see [below](#configuration-and-options).

#### More background on the data setup
This is a file format for medical imaging that keeps lots of related things together--
in our case, it's mostly useful because it lets us store our image and label
together in one file which will guarantee we won't do anything silly like accidentally
match up the wrong label to an image.

The labels live in the `1Felix and Rich make models/Training dataset Tiffs` directory here.
The original images live elsewhere - 3D TIFFs are in `DATABASE/uCT/Wahab_clean_dataset/TIFS/`.

There are also some 2D TIFFs in the `DATABASE/uCT/Wahab_clean_dataset/` directory somewhere, but
I would recommend dealing with DICOMs wherever possible.
If you have to deal with TIFF files try to read/write 3D TIFFs where possible instead of dealing
with lots of 2d TIFFs, because it's much faster to read one big file than many small ones.

### Training the model
Train the model by running the `scripts/train_model.py` script.
```
PYTHONPATH=$(pwd) python scripts/train_model.py
```

This will create some plots showing the training process-- the training and validation loss,
an example of the training data, an example of inference on some testing data, etc.
These are saved to the `train_output/` directory.

On an old-ish NVIDIA A6000 GPU (48GB RAM) it took me around 20 minutes to train a model on 27 complete jaws,
but this is subject to change as we get more data and as I mess with everything.

### Running inference
Perform inference using `inference_example.py`.

```
PYTHONPATH=$(pwd) python scripts/train_model.py
```

### Going further
You can turn the voxels from the inference into a mesh by learning how to do that

### Configuration and options
I've tried to make all the assumptions and choices as transparent as possible in this project; because if you don't
then soon the complexity quickly outruns what you can keep in your head and things end up going wrong, and also
because I thought it would be fun.
As such, most of the options and assumptions are encoded explicitly in one of two configuration files, `config.yml`
and `userconf.yml`.

You probably don't need to edit anything in `config.yml`; it's for things that would otherwise be hard-coded in the
source files (e.g. file locations that we don't expect to change, like things stored on the RDSF).

`userconf.yml` contains things that you might want to edit-- things like model hyperparameters (batch size, number
of epochs etc.) but also the name of the model architecture, the transforms to use and which DICOMs to use as testing
or validation data.

If you're running this code on your own computer, you'll want to change certain things like where the RDSF is mounted.

## For Developers
i am the developer

### CI

### Tests

### Linting, formatting and typing

## Footnotes
[^1]: Kawafi, A., 2023. Quantitative 3D Bioimaging Using Machine Learning (Doctoral dissertation, University of Bristol).t:
