# zebrafish_jaw_segmentation
Segmentation of Zebrafish Jawbones from uCT scans

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

The jaw segmentation is heavily based on the work performed by Wahab Kawafi in his PhD thesis[^1].

The segmentation model is implemented in `pytorch`, using the `monai` `AttentionUnet` architecture.

## Usage
### Hardware
You'll need an nvidia GPU to make this code work; a lot of it relies on CUDA. The right libraries should be installed by `uv`.

If you don't have cuda installed at all, you'll need to do that first-- on linux, this will be something like
`sudo apt install nvidia-cuda-toolkit`.


### Environment
The python environment here is managed using `uv`; run scripts with e.g.:

```
uv run python my_cool_script.py
```

To create the python environment:
```
uv sync
```
> [!IMPORTANT]
> If you haven't done so already, you will need to [install `uv`](https://docs.astral.sh/uv/getting-started/installation/) first

If you're making any changes, you may want to additionally install the dev dependencies (for linting, testing etc.)
```
uv sync --dev
```

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

On an old-ish NVIDIA A6000 GPU (48GB RAM) it took me around 13 minutes to train a model for 100 epochs on 43 images.

### Running inference
Perform inference using `inference_example.py`.

```
PYTHONPATH=$(pwd) python scripts/train_model.py
```

#### Running inference on a new fish
I've hard coded lots of things which makes it slightly irritating to run the inference on a new fish; I haven't got around
to making a nicer user interface for it yet.

To run the inference on a new fish:
 - Convert the CT scan to a 3d TIF if it isn't already, with filename `<n>.tif`
 - Copy it to the "ct_scan_dir" specified in `config.yml`.
 - Add the centre of the jaw to the dict in `fishjaw.inference.read.crop_lookup`, using `<n>` as the key
 - run `scripts/inference.py` with `<n>` as a CLI argument

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
I've used GitHub Actions to run the CI. It doesn't do all that much.

### Linting, formatting and typing
This is the main QA type stuff that I've set up-- linting with `pylint`, format checking with `black`.
I also intended to set up type checking with `mypy`, but I haven't done that yet.

### Tests
These were basically an afterthought.
Tests are in `fishjaw/tests/`.
There are unit tests (fast), integration tests (also fast, but rely on interactions between things, reading files etc.)
and system tests (slow and test a lot functionality at once).

## Footnotes
[^1]: Kawafi, A., 2023. Quantitative 3D Bioimaging Using Machine Learning (Doctoral dissertation, University of Bristol).t:
