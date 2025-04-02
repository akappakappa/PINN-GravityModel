% Load data
PlanesTrj = dlarray(table2array(readtable("src/preprocessing/datastore/metrics/PlanesTrj.csv")), 'BC');
PlanesAcc = dlarray(table2array(readtable("src/preprocessing/datastore/metrics/PlanesAcc.csv")), 'BC');
PlanesPot = dlarray(table2array(readtable("src/preprocessing/datastore/metrics/PlanesPot.csv")), 'BC');

RandomTrj = dlarray(table2array(readtable("src/preprocessing/datastore/metrics/RandomTrj.csv")), 'BC');
RandomAcc = dlarray(table2array(readtable("src/preprocessing/datastore/metrics/RandomAcc.csv")), 'BC');
RandomPot = dlarray(table2array(readtable("src/preprocessing/datastore/metrics/RandomPot.csv")), 'BC');

% Load Network
net = load("src/training/net.mat");

% Calculate Planes metric
PlanesMetric = dlfeval(@planes, net, PlanesTrj, PlanesAcc, PlanesPot);
fprintf("Planes metric: %f\n", PlanesMetric);

% Calculate Generalized metric
GeneralizedMetric = dlfeval(@generalized, net, RandomTrj, RandomAcc, RandomPot);
fprintf("Generalized metric: %f\n", GeneralizedMetric);