% Load data
PlanesTrj = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesTrj.csv"), 'BC');
PlanesAcc = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesAcc.csv"), 'BC');
PlanesPot = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesPot.csv"), 'BC');

RandomTrj = dlarray(readmatrix("src/preprocessing/datastore/metrics/RandomTrj.csv"), 'BC');
RandomAcc = dlarray(readmatrix("src/preprocessing/datastore/metrics/RandomAcc.csv"), 'BC');
RandomPot = dlarray(readmatrix("src/preprocessing/datastore/metrics/RandomPot.csv"), 'BC');

% Load Network
net = load("src/training/net.mat").net;

% Compute metrics
PlanesMetric      = dlfeval(@planes     , net, PlanesTrj, PlanesAcc, PlanesPot);
fprintf("Planes metric: %f\n"     , PlanesMetric     );
GeneralizedMetric = dlfeval(@generalized, net, RandomTrj, RandomAcc, RandomPot);
fprintf("Generalized metric: %f\n", GeneralizedMetric);