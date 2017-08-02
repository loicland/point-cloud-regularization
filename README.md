## point-cloud-regularization

A structured optimization framework for spatial regularization and segmentation of point clouds, with Matlab interface
Loic Landrieu 2017


# Regularization: 

![regularization](https://user-images.githubusercontent.com/1902679/28877706-76ef2fa8-779d-11e7-84a3-6288594c8b73.png)
![segmentation](https://user-images.githubusercontent.com/1902679/28877709-76f86c08-779d-11e7-8b27-41a0ec4e3187.png)

Based on:

A structured regularization framework for spatially smoothing semantic labelings of 3D point clouds.
Landrieu, L., Raguet, H., Vallet, B., Mallet, C., & Weinmann, M. (2017).

This framework propose a set of methods for spatialy regularizing semantic labelings on a point cloud.
As mentioned in the paper above, 4 fidelity functions and 3 regularizers are proposed.


# Segmentation:

Based on:

Weakly supervised segmentation-aided classification of urban scenes from 3D LiDAR point clouds.
Guinard, S., & Landrieu, L. In ISPRS 2017

Fast segmentation of point clouds with L0-cut pursuit.

## DEPENDENCIES:

CUT PURSUIT : https://github.com/loicland/cut-pursuit

PFDR : (to come very soon)

ALPHA-EXPANSION : http://vision.ucla.edu/~brian/gcmex.html

LOOPY BELIEF PROPAGATION : http://www.cs.ubc.ca/~schmidtm/Software/UGM.html

All those dependencies are optional, but access to the corresponding regularization are dependant on which ones are installed. If you chose not to install some of those libraries, some code commenting might be necessary.

The data compressed files needs to be dezipped.
