End State
----
Checklist for handoverable state:
 [ ] New clean repo

 [ ] Functionality:
     [ ] Libraries
         [ ] Add the tests all in
         [ ] Add library functions that make the tests pass
         [ ] Don't add any other non-tested bits - these should come out when we add the scripts below...
     [ ] Create DICOMs
         [ ] create_dicoms.py
     [ ] Train the segmentation model
         [ ] train_model.py
     [ ] Reproduce the things in the paper
         [ ] Small facts/summaries
              [ ] arch_summary.py 
              [ ] boring_scripts/training_summary_stats.py
              [ ] boring_scripts/find_occupancy.py - find the size of the jaw
              [ ] boring_scripts/rotating_truth_images.py - for making animations of the jaws for slides
         [ ] Mainline Analysis
             [ ] inference_example.py
             [ ] compare_segmentations.py
             [ ] compare_slices.py - plot the slices in the fig
             [ ] cubes.py - plot the cube illustration
             [ ] plot_3d_rearjaw.py - plot the jaw joint images
             [ ] train_models.sh - shell script to train all the models
             [ ] repeat_training_summary.py - summarise the result

     [ ] Train the jaw locator
         [ ] train_jaw_locating_model.py
     [ ] Example for running inference on a lot jaws
         [ ] scripts/pipeline/segment_jaws.py
         [ ] scripts/pipeline/plot_slices.py
         [ ] scripts/pipeline/plot_3d.py
         [ ] scripts/pipeline/shape_plots.py
     [ ] Example for extracting greyscale stats + roughly finding muscle attachements
         [ ] scripts/pipeline/greyscale_plots.py - simple greyscale plots
         [ ] muscle_attachment_analysis/find_muscle_attachments.ipynb
     [ ] Example fine tuning
         [ ] Perhaps based on quadrate_transfer_learning.py
     [ ] Toy/not good feature selection script, with its own pyproject.toml
         [ ] scripts/notebooks/feature_extraction/feature_extraction.py
         [ ] scripts/notebooks/feature_extraction/pyproject.toml
     [ ] Extraneous scripts
         [ ] plot_dicoms.py
         [ ] plot_train_data.py
         [ ] ablate_attention.py
         [ ] mesh.py
         [ ] explore_hyperparams.py
         [ ] plot_hyperparams.py
         [ ] find_rear_jaw_centres.py - used to make the jaw_centres.csv spreadsheet
         [ ] plot_cropped.py - for checking I've got the right ROI, and that we have the right slices for the rear jaw images
     [ ] Make a pipeline script type thing


[ ] Documentation
     [ ] For the above
         [ ] Justify having DICOMs - speed, and to keep labels with images
         [ ] The segmentation model
         [ ] Create figures from the paper
         [ ] We need jaw locator since the model takes a cropped fish head
         [ ] Our model is useful when we run it on lots of scans - this is that
         [ ] Justification for looking at greyscale
     [ ] Future work doc
         [ ] Mention using Shapeworks to analyse shape
         [ ] Write about feature selection - why, what we're doing, how it can be improved, my planned task using the mutation to discriminate based on features
         [ ] Write about registration of attachment sites
     [ ] Useful README with how to install, run, basic justification etc.

[ ] Handover
     [ ] Make slides on the motivation + results
     [ ] Get the JGI to try and run it
     [ ] Get someone in biomed to try and run it, in principle


## New clean repo
 [ ] Make a new repo
 [ ] Init a project - with uv
 [ ] Add the right dependencies
 [ ] Init a script for creating DICOMs
 [ ] 

