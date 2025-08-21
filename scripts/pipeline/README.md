Pipeline
====

Scripts for complete inference pipelines - from CT scan data through
to some sort of output

 - `segment_jaws.py` - segment all the jaws in Wahab's 3D tif dir, and save the output somewhere that we can use later
 - `plot_slices.py` - plot slices through the cropped CT scans/inference masks
 - `plot_3d.py` - plot projections of the jaw in 3D, colour-coded by the greyscale value of that pixel
 - `shape_plots.py` - simple analysis of the shape of the jaws - i.e. things that can be inferred from the mask only.
 - `greyscale_plots.py` - simple analysis of the greyscale content of the jaws (so far just hists, but more (boxplots?) should come soon).
