Data
====

This is where some utilities to do with data live.
It's also where some files will get created - notably,
the model weights once they're trained.

`jaw_centres.csv`
----
This is a table holding information about where the jaw in in each scan.
We need this because we crop out the jaw region when we train the model-
we don't want to show the model the entire CT scan every time,
we just need to show it the region around the jaw.

There's two different cases here:
### Whole Jaw
Most of the training data contains the whole jaw.

For memory efficiency, and to help the model learn, we crop
out a region containing the jaw and show this region to the
model.
The only thing that matters for this cropping is that the region contains
the jaw (in fact, it's not strictly necessary that the whole jaw is contained in the region;
all we require is good representation of the jaw shape across
the training data, but the best use of the data will contain
the whole of each jaw).
This is unlike the case below, where we need to be more careful.

To address this, `jaw_centres.csv` contains the XYZ location of the centre of each jaw,
which is used as the centre of the cube to crop around.
This centre was found by hand-- I opened each mask in ImageJ,
found which slice contained the start and end of the labelled region, and found the slice in the middle of the two.
This gives us the Z co-ordinate.
Then I found the XY co-ordinate in the middle of this slice
by eye.


### Rear Jaw only
A small subset of the training data only contains the rear portion of the jaw.
This is intended to make the model better at recognising
tissue in the difficult region where there's a lot of
different tissues of different densities.

The difficulty here is that we don't want to show the model any unlabelled jaw--
this would be actively detrimental and would harm the model's ability to distinguish between the jaw and background.
The partially labelled jaw images contain some unlabelled jaw
tissue (this is what we don't want to show the model - we
only want it to see labelled jaw and unlabelled background),
which must be cropped out before it is passed to the model.

We thus want the region of interest to extend from somewhere
in the fish's body to the last slice containing labelled jaw,
and no further.

To address this, `jaw_centres.csv` contains the XYZ location
of the back of the jaw (so maybe it should have a different
filename, but I don't want to change it right now).
When we crop out these rear jaw images, we will include the
last Z slice that has been labelled.

Indexing from 0, cropping up to the `Z` in this table (e.g.
by using `array[z-width:z]` if `array` is a `numpy` array)
will give us a cube which extends up to the last labelled
slice and no further.

An illustration of this can be made with the`scripts/plot_crops.py`
script.
Pass this script the path to a DICOM file (as created by
`scripts/create_dicoms.py`) and it will plot the first and
last slices, the ones immediately before and after that,
and some in between.
We'd expect the last slice for the cropped DICOM to contain a
lot of labelled jaw, and the one after that to contain no
labelling.

For example:

add an image of that

## TODO make sure this is true then delete this line from the README. i bet ill forget this hehe
