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
GeneralizationTrj        = cat(2, GeneralizationTrj_0_1, GeneralizationTrj_1_10, GeneralizationTrj_10_100);
GeneralizationAcc        = cat(2, GeneralizationAcc_0_1, GeneralizationAcc_1_10, GeneralizationAcc_10_100);
GeneralizationPot        = cat(2, GeneralizationPot_0_1, GeneralizationPot_1_10, GeneralizationPot_10_100);

SurfaceTrj               = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfaceTrj.csv"              ), 'BC');
SurfaceAcc               = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfaceAcc.csv"              ), 'BC');
SurfacePot               = dlarray(readmatrix("src/preprocessing/datastore/metrics/SurfacePot.csv"              ), 'BC');

% Load Network
net = load("src/training/net.mat").net;

% Remove points inside the asteroid
[PlanesTrj, PlanesAcc, PlanesPot] = removeInside(PlanesTrj, PlanesAcc, PlanesPot);

% Compute metrics
% ---------------------------------- | Preset function ----- | NN | Trajectory Data ------- | Acceleration Data ----- | Potential Data -------- |
mpePlanesMetric                = dlfeval(@presets.mpeLoss, net, PlanesTrj               , PlanesAcc               , PlanesPot               );
mePlanesMetric                 = dlfeval(@presets.meLoss , net, PlanesTrj               , PlanesAcc               , PlanesPot               );
mpeGeneralizationMetric_0_1    = dlfeval(@presets.mpeLoss, net, GeneralizationTrj_0_1   , GeneralizationAcc_0_1   , GeneralizationPot_0_1   );
meGeneralizationMetric_0_1     = dlfeval(@presets.meLoss , net, GeneralizationTrj_0_1   , GeneralizationAcc_0_1   , GeneralizationPot_0_1   );
mpeGeneralizationMetric_1_10   = dlfeval(@presets.mpeLoss, net, GeneralizationTrj_1_10  , GeneralizationAcc_1_10  , GeneralizationPot_1_10  );
meGeneralizationMetric_1_10    = dlfeval(@presets.meLoss , net, GeneralizationTrj_1_10  , GeneralizationAcc_1_10  , GeneralizationPot_1_10  );
mpeGeneralizationMetric_10_100 = dlfeval(@presets.mpeLoss, net, GeneralizationTrj_10_100, GeneralizationAcc_10_100, GeneralizationPot_10_100);
meGeneralizationMetric_10_100  = dlfeval(@presets.meLoss , net, GeneralizationTrj_10_100, GeneralizationAcc_10_100, GeneralizationPot_10_100);
mpeGeneralizationMetric        = dlfeval(@presets.mpeLoss, net, GeneralizationTrj       , GeneralizationAcc       , GeneralizationPot       );
meGeneralizationMetric         = dlfeval(@presets.meLoss , net, GeneralizationTrj       , GeneralizationAcc       , GeneralizationPot       );
mpeSurfaceMetric               = dlfeval(@presets.mpeLoss, net, SurfaceTrj              , SurfaceAcc              , SurfacePot              );
meSurfaceMetric                = dlfeval(@presets.meLoss , net, SurfaceTrj              , SurfaceAcc              , SurfacePot              );

fprintf("\n### Mean Percent Error (MPE) ###\n");
fprintf("Planes metric                   : %f\n", mpePlanesMetric               );
fprintf("Generalization metric [0R:1R]   : %f\n", mpeGeneralizationMetric_0_1   );
fprintf("Generalization metric [1R:10R]  : %f\n", mpeGeneralizationMetric_1_10  );
fprintf("Generalization metric [10R:100R]: %f\n", mpeGeneralizationMetric_10_100);
fprintf("Generalization metric [0R:100R] : %f\n", mpeGeneralizationMetric       );
fprintf("Surface metric                  : %f\n", mpeSurfaceMetric              );

fprintf("\n### Mean Error (ME) ###\n");
fprintf("Planes metric                   : %f\n", mePlanesMetric               );
fprintf("Generalization metric [0R:1R]   : %f\n", meGeneralizationMetric_0_1   );
fprintf("Generalization metric [1R:10R]  : %f\n", meGeneralizationMetric_1_10  );
fprintf("Generalization metric [10R:100R]: %f\n", meGeneralizationMetric_10_100);
fprintf("Generalization metric [0R:100R] : %f\n", meGeneralizationMetric       );
fprintf("Surface metric                  : %f\n", meSurfaceMetric              );



function [Trj, Acc, Pot] = removeInside(Trj, Acc, Pot)
    if   ~isfile("src/test/shape.mat")
         shape = readObj("src/data/Model/eros_shape_200700.obj");
         max_extent = max(max(abs(shape.v)));  
         shape.v = shape.v / max_extent;
         save("shape", "shape");
    else load("src/test/shape.mat");
    end
         trj = extractdata(Trj);
         trj = trj';
         inside = inpolyhedron(shape.f.v, shape.v, trj);
         inside = inside';
         Trj = Trj(:, ~inside);
         Acc = Acc(:, ~inside);
         Pot = Pot(:, ~inside);
end