Boring Scripts
====

Scripts that I only ran once (or a few times) to check stuff, that you don't really
need to care about or run again - I'm just keeping them here just in case


Contents
----
- `voxels.py`: create voxel plots showing the labels specified
- `check_loss.py`: check that the loss function behaves as you might expect,
                 including how it behaves at chance and perfect performance.
                 So that the expected loss is 0.0 at perfect performance and
                 $\frac{1}{1+\frac{1}{2}\alpha + \frac{1}{2}\beta}$ at chance performance,
                 This script doesn't use a sigmoid or softmax activation fcn.
                 Creates a plot named `chance_loss.png`.
- `find_rear_jaw_centres.py`: find the ZXY locations of the jaw in of the rear jaw dataset. We need this
                              because we want to crop our jaw from the last labelled slice backwards (towards
                              the tail) so that our cropped out regions of interest don't contain any unlabelled
                              jaw voxels. This script makes some decisions and crops the jaws.
- `plot_cropped.py`: check that the cropping code gives us the correct regions.
                   This is intended to be a check that the cropping code correctly
                   gives us the correct slices for the rear jaw DICOMs, since it's
                   important that there's no unlabelled jaw in the training data.
- `lines_of_code.py`: plot the total lines of code in this respository over time. Just for fun!!!!
- `find_occupancy.py`: find how much of each DICOM is occupied by jaw voxels, for both the cropped and uncropped DICOMs.