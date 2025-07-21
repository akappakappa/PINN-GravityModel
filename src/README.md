# Code preview
The code entry-point is [main](main.m), which conditionally launches all the routines.

## [Data Extraction](data/runData.m)
Extracts python-pickle data from the [dataset folder](data/Trajectories/), generated using the [GravNN](https://github.com/MartinAstro/GravNN) library, then saves the results to file.

**NOTE** set [DO_DATA_EXTRACTION](main.m#L15) to `false` to skip the execution of this file, if you don't want to install the required Python dependencies, as pre-extracted data is included in the repo.

## [Preprocessing](preprocessing/runPreprocessing.m)
Computes $\mu=GM$ parameter for the Low-Fidelity Analytical Model that will be fused to the Neural Network predictions, performs non-dimensionalization, finally splits the dataset with a ratio 99:1 into training:validation sets.

**NOTE** set [DO_PREPROCESSING](main.m#L16) to `false` to skip the execution of this file, as pre-extracted data is included in the repo.

## [Training](training/runTraining.m)
Contains the main training loop and [presets](training/+presets) for custom layers, losses, network structures, training options, and learnrate schedulers.  
Specifically, the loss function leverages *autodiff* to compute the gradients of the predicted potential wrt input Cartesian coordinates, emebdding the physical constraint: $\mathbf{a}=-\nabla U$.  
$L=\frac{1}{N}\sum_{i=0}^N\left(\left\|-\nabla\hat{U}(\mathbf{x}_i)-\mathbf{a}_i\right\|+\frac{\left\|-\nabla\hat{U}(\mathbf{x}_i)-\mathbf{a}_i\right\|}{\left\|\mathbf{a}_i\right\|}\right)$  
where $\nabla\hat{U}(\mathbf{x}_i)$ represents the predicted acceleration and $\mathbf{a}_i$ is the true acceleration.

## [Testing](test/runTest.m)
Computes metrics, plots and saves relevant figures comparing performance to the Polyhedral model.

All metrics compute a Mean Percent Error (MPE)  
$L=\frac{1}{N}\sum_{i=0}^N\frac{\left\|\hat{\mathbf{a}}_i-\mathbf{a}_i\right\|}{\left\|\mathbf{a}_i\right\|}\times100$

### [Generalization](test/plot/plotGeneralization.m)
500 samples per unit of radius in 0:100R (50'000 total), evaluated separately depending on the altitude as **interior**, **exterior** and **extrapolation** (0:1R, 1:10R, 10:100R respectively).

### [Planes](test/plot/plotPlanes.m)
Samples along the three Cartesian planes XY, XZ, YZ between [-5R, 5R] displaced in 200*200 grids.

### [Surface](test/plot/plotSurface.m)
~200'000 samples on the surface of the asteroid, one in the centre of each facet of the shape model.