Scripts
====
Some scripts


Setup
----
`create_dicoms.py`: create DICOM files for each CT scan - label pair
`plot_dicoms.py`: create plots visualising these DICOM files

Training Models
----
`train_model.py`: train the model

Hyperparameter Tuning
----
`explore_hyperparams.py`: train lots of models with different hyperparameters, to see what's best
                          This script isn't particularly good or robust - most of the options are defined
                          by the config `dict` within this script, but there are some other parameters and
                          extra things (e.g. the transformations that get applied) that are hard-coded in other
                          places.
`plot_hyperparams.py`: plot the result of the hyperparameter tuning

Other Stuff
----
`mesh.py`: example showing the conversion from tiff (which is what the segmentation model creates) to a mesh
    [ ] TODO make this useful
`arch_summary.py`: summarise the architecture of the model (at the moment just prints the feature map sizes)