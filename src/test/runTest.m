% Load data
PlanesTrj         = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesTrj.csv"        ), 'BC');
PlanesAcc         = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesAcc.csv"        ), 'BC');
PlanesPot         = dlarray(readmatrix("src/preprocessing/datastore/metrics/PlanesPot.csv"        ), 'BC');

GeneralizationTrj = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationTrj.csv"), 'BC');
GeneralizationAcc = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationAcc.csv"), 'BC');
GeneralizationPot = dlarray(readmatrix("src/preprocessing/datastore/metrics/GeneralizationPot.csv"), 'BC');

SurfaceTrj        = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfaceTrj.csv"       ), 'BC');
SurfaceAcc        = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfaceAcc.csv"       ), 'BC');
SurfacePot        = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfacePot.csv"       ), 'BC');

% Load Network
net = load("src/training/net.mat").net;

% Compute metrics
PlanesMetric         = dlfeval(@presets.planes        , net, PlanesTrj        , PlanesAcc        , PlanesPot        );
fprintf("Planes metric         : %f\n", PlanesMetric        );

GeneralizationMetric = dlfeval(@presets.generalization, net, GeneralizationTrj, GeneralizationAcc, GeneralizationPot);
fprintf("Generalization metric : %f\n", GeneralizationMetric);

SurfaceMetric        = dlfeval(@presets.surface       , net, SurfaceTrj       , SurfaceAcc       , SurfacePot       );
fprintf("Surface metric        : %f\n", SurfaceMetric       );