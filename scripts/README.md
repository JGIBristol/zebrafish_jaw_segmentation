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
`plot_hyperparams.py`: plot the result of the hyperparameter tuning

Other Stuff
----
`mesh.py`: example showing the conversion from tiff (which is what the segmentation model creates) to a mesh
    [ ] TODO make this useful