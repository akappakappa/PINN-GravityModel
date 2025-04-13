# Gravity Model using PINNs
This Master's Thesis project is about the application of Physics-Informed Neural Networks (PINNs) to the gravity model problem for celestial bodies.  
MATLAB R2024b is used for the implementation of the project.

### TODO
- [x] Generate the dataset for training / testing.
- [x] Data extraction
- [x] Preprocessing
- [x] Training
- [ ] Metrics
- [x] Going fast on GPU!
- [ ] Expand project ...

## Code preview
The code entry-point is [main](./src/main.m), which simply conditionally launches the following routines:
1. [Data Extraction](./src/data/runData.m) extracts python-pickle data from the [dataset folder](./src/data/Trajectories/), generated using the [GravNN](https://github.com/MartinAstro/GravNN) library for consistency, then saves the results to file. \
   **NOTE** set [DO_DATA_EXTRACTION](./src/main.m#L10) to `false` to skip the execution of this file, if you don't want to install the required Python dependencies, as pre-extracted data is included in the repo.
2. [Preprocessing](./src/preprocessing/runPreprocessing.m) performs the following:
    - Computes parameters for the Low-Fidelity Analytical Model that will be fused to the Neural Network predictions ($\mu$ and $e$).
    - Non-dimensionalizes the dataset.
    - Splits the dataset with a ratio 9:1 into training:validation sets.
3. [Training](./src/training/runTraining.m) contains the main loop:
    - [Network structure](./src/training/+presets/+network/PINN_GM_III.m), with [custom layers](./src/training/+presets/+layer/) for feature engineering, prediction adjustments and model fusion. \
    $\hat{U}(r)=w_{NN}(w_{F}U_{LF}(r)+\hat{U}_{NN}(r))+w_{BC}U_{LF}(r)$ \
    where $w$ are transition functions $H(r,s,ref)=\frac{1+tanh(s(r-ref))}{2}$ \
    and $w_{NN}=1-w_{BC}$
    - [Loss function](./src/training/+presets/+loss/PINN_GM_III.m) that combines RMS and MPE losses. \
    $L_{RMS+MPE}(\theta)=\frac{1}{N}\sum_{i=0}^N\left(\sqrt{\left|-\nabla\hat{U}(x_i|\theta)-a_i\right|^2}+\frac{\left|-\nabla\hat{U}(x_i|\theta)-a_i\right|}{|a_i|}\right)$ \
    where $\nabla\hat{U}(x_i|\theta)$ is the differentaited network potential and $a_i$ is the true acceleration.
4. [Test Metrics](./src/test/runTest.m) **TODO** implements the following metrics:
    - [x] Planes
    - [x] Generalized
    - [x] Surface
    - [ ] Trajectory NEAR (unable to generate it, missing files from [GravNN](https://github.com/MartinAstro/GravNN))
 
## Authors
Computer Engineering @ Unversity Of Padua, Italy:
- [Andrea Valentinuzzi](github.com/akappakappa)
- [Giovanni Brejc](github.com/Govawi)