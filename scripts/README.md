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

Some others that you probably won't run at all, because they're basically not interesting
- `check_loss.py`: check that the loss function behaves as you might expect,
                 including how it behaves at chance and perfect performance.
                 So that the expected loss is 0.0 at perfect performance and
                 $\frac{1}{1+\frac{1}{2}\alpha + \frac{1}{2}\beta}$ at chance performance,
                 This script doesn't use a sigmoid or softmax activation fcn.
                 Creates a plot named `chance_loss.png`.
- `plot_cropped.py`: check that the cropping code gives us the correct regions.
                   This is intended to be a check that the cropping code correctly
                   gives us the correct slices for the rear jaw DICOMs, since it's
                   important that there's no unlabelled jaw in the training data.
