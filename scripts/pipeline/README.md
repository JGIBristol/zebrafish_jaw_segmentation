Pipeline
====
Analysis of the jaw segmentations once I've done a bunch of them

1. `segment_jaws.py` - run the jaw localisation and segmentation on lots of jaws from the RDSF, and save the cropped images and predicted masks.
2. `shape_plots.py` - plot age vs volume, length etc. for a quick summary of the metadata.
3. `greyscale_plots.py` - plot summaries of greyscale stats, like how their averages change with age/length.
4. `plot_slices.py` - plot slices through the segmentations
5. `plot_3d.py` - plot projections of the jaw in 3d to see its shape