# FeaturedMind #

This project is meant to be a workbench for experiments with deep learning based image segmentation setups. 

The main purpose is to create a model which is able to detect various features in a high resolution image:

* A specimen which should be scanned, for instance a sheet metal part.
* A fixture which might be in the optical path between scanner and specimen.

This is a proof-of-concept showing that a task like this is possible with reasonable loss and especially recall. 
If working it might be possible to measure real life setups with fixture parts blocking a scanning sensor and decide which parts of the scanned image has to be masked to get a meaningful polygonization nevertheless.

![Generated specimen and fixture with artificial background and resulting mask](images/readme_specimen_and_mask.png "Generated specimen and fixture with artificial background and resulting mask")