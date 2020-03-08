## point-cloud-regularization

A structured optimization framework for spatial regularization and segmentation of point clouds, with Matlab interface
Loic Landrieu 2017


# Regularization: 

![regularization](https://user-images.githubusercontent.com/1902679/28877990-49e1662e-779e-11e7-96cb-8f49e4700a09.png)

Based on:

A structured regularization framework for spatially smoothing semantic labelings of 3D point clouds.
Landrieu, L., Raguet, H., Vallet, B., Mallet, C., & Weinmann, M. (2017).

This framework propose a set of methods for spatialy regularizing semantic labelings on a point cloud.
As mentioned in the paper above, 4 fidelity functions and 3 regularizers are proposed.


# Segmentation:

![segmentation](https://user-images.githubusercontent.com/1902679/28877979-44c9c726-779e-11e7-87d6-75cf853f9622.png)

Based on:

Weakly supervised segmentation-aided classification of urban scenes from 3D LiDAR point clouds.
Guinard, S., & Landrieu, L. In ISPRS 2017

Fast segmentation of point clouds with L0-cut pursuit.

## DEPENDENCIES:

CUT PURSUIT : https://github.com/loicland/cut-pursuit

PFDR : From https://github.com/1a7r0ch3

ALPHA-EXPANSION / GCMEX : https://github.com/shaibagon/GCMex

LOOPY BELIEF PROPAGATION : http://www.cs.ubc.ca/~schmidtm/Software/UGM.html

All those dependencies are optional, but access to the corresponding regularization are dependant on which ones are installed. If you chose not to install some of those libraries, some code commenting might be necessary.

The data compressed files needs to be dezipped. All credits goes to http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/ for the data.

## RUNNING THE CODE:

Run the lines from `configure.m` corresponding to the method you are interested to try.

Follow `benchmark.m` to see examples of calls.
