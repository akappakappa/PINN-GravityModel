% Load data
PlanesTrj                = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesTrj.csv"               ), 'BC');
PlanesAcc                = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesAcc.csv"               ), 'BC');
PlanesPot                = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesPot.csv"               ), 'BC');

GeneralizationTrj_0_1    = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationTrj_0_1.csv"   ), 'BC');
GeneralizationAcc_0_1    = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationAcc_0_1.csv"   ), 'BC');
GeneralizationPot_0_1    = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationPot_0_1.csv"   ), 'BC');
GeneralizationTrj_1_10   = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationTrj_1_10.csv"  ), 'BC');
GeneralizationAcc_1_10   = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationAcc_1_10.csv"  ), 'BC');
GeneralizationPot_1_10   = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationPot_1_10.csv"  ), 'BC');
GeneralizationTrj_10_100 = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationTrj_10_100.csv"), 'BC');
GeneralizationAcc_10_100 = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationAcc_10_100.csv"), 'BC');
GeneralizationPot_10_100 = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationPot_10_100.csv"), 'BC');

SurfaceTrj               = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfaceTrj.csv"              ), 'BC');
SurfaceAcc               = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfaceAcc.csv"              ), 'BC');
SurfacePot               = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfacePot.csv"              ), 'BC');

% Load Network
net = load("src/training/net.mat").net;

% Planes Metric
PlanesMetric         = dlfeval(@presets.planes        , net, PlanesTrj        , PlanesAcc        , PlanesPot        );
fprintf("Planes metric         : %f\n", PlanesMetric        );

% Generalization Metric
GeneralizationMetric_0_1 = dlfeval(@presets.generalization, net, GeneralizationTrj_0_1, GeneralizationAcc_0_1, GeneralizationPot_0_1);
fprintf("Generalization metric [0R:1R]: %f\n", GeneralizationMetric_0_1);

GeneralizationMetric_1_10 = dlfeval(@presets.generalization, net, GeneralizationTrj_1_10, GeneralizationAcc_1_10, GeneralizationPot_1_10);
fprintf("Generalization metric [1R:10R]: %f\n", GeneralizationMetric_1_10);

GeneralizationMetric_10_100 = dlfeval(@presets.generalization, net, GeneralizationTrj_10_100, GeneralizationAcc_10_100, GeneralizationPot_10_100);
fprintf("Generalization metric [10R:100R]: %f\n", GeneralizationMetric_10_100);

GeneralizationTrj = cat(1, GeneralizationTrj_0_1, GeneralizationTrj_1_10, GeneralizationTrj_10_100);
GeneralizationAcc = cat(1, GeneralizationAcc_0_1, GeneralizationAcc_1_10, GeneralizationAcc_10_100);
GeneralizationPot = cat(1, GeneralizationPot_0_1, GeneralizationPot_1_10, GeneralizationPot_10_100);

GeneralizationMetric = dlfeval(@presets.generalization, net, GeneralizationTrj, GeneralizationAcc, GeneralizationPot);
fprintf("Generalization metric [0R:100R]: %f\n", GeneralizationMetric);

% Surface Metric
SurfaceMetric        = dlfeval(@presets.surface       , net, SurfaceTrj       , SurfaceAcc       , SurfacePot       );
fprintf("Surface metric        : %f\n", SurfaceMetric       );