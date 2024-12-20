Scripts
====
These are the shiny face of the code here-- the scripts that actually do stuff, like train the model or plot things.

These scripts are all run from the root of this repository via the command line as follows:

```
PYTHONPATH=$(pwd) python scripts/create_dicoms.py
```

Contents
----
Roughly in the order that you might run them:
- `create_dicoms.py`: create DICOM files for each CT scan-label pair.
- `plot_dicoms.py`: create plots visualising these DICOM files.
- `plot_train_data.py`: create plots showing the model's training data.
- `explore_hyperparams.py`: run hyperparameter tuning
- `plot_hyperparams.py`: visualise the result of the hyperparameter tuning
- `train_model.py`: train a model
- `arch_summary.py`: summarise the architecture of the model
- `inference_example.py`: run inference on some data
- `mesh.py`: create and visualise a mesh from the inference
