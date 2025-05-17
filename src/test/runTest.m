% This script tests the performance of the trained model.

headless        = batchStartupOptionUsed;
metricsFolder   = "src/preprocessing/datastore/metrics/";

% Load data
PlanesTrj                = dlarray(readmatrix(metricsFolder + "PlanesTrj.csv"               ), 'BC');
PlanesAcc                = dlarray(readmatrix(metricsFolder + "PlanesAcc.csv"               ), 'BC');
PlanesPot                = dlarray(readmatrix(metricsFolder + "PlanesPot.csv"               ), 'BC');

GeneralizationTrj_0_1    = dlarray(readmatrix(metricsFolder + "GeneralizationTrj_0_1.csv"   ), 'BC');
GeneralizationAcc_0_1    = dlarray(readmatrix(metricsFolder + "GeneralizationAcc_0_1.csv"   ), 'BC');
GeneralizationPot_0_1    = dlarray(readmatrix(metricsFolder + "GeneralizationPot_0_1.csv"   ), 'BC');
GeneralizationTrj_1_10   = dlarray(readmatrix(metricsFolder + "GeneralizationTrj_1_10.csv"  ), 'BC');
GeneralizationAcc_1_10   = dlarray(readmatrix(metricsFolder + "GeneralizationAcc_1_10.csv"  ), 'BC');
GeneralizationPot_1_10   = dlarray(readmatrix(metricsFolder + "GeneralizationPot_1_10.csv"  ), 'BC');
GeneralizationTrj_10_100 = dlarray(readmatrix(metricsFolder + "GeneralizationTrj_10_100.csv"), 'BC');
GeneralizationAcc_10_100 = dlarray(readmatrix(metricsFolder + "GeneralizationAcc_10_100.csv"), 'BC');
GeneralizationPot_10_100 = dlarray(readmatrix(metricsFolder + "GeneralizationPot_10_100.csv"), 'BC');
GeneralizationTrj        = cat(2, GeneralizationTrj_0_1, GeneralizationTrj_1_10, GeneralizationTrj_10_100);
GeneralizationAcc        = cat(2, GeneralizationAcc_0_1, GeneralizationAcc_1_10, GeneralizationAcc_10_100);
GeneralizationPot        = cat(2, GeneralizationPot_0_1, GeneralizationPot_1_10, GeneralizationPot_10_100);

SurfaceTrj               = dlarray(readmatrix(metricsFolder + "SurfaceTrj.csv"              ), 'BC');
SurfaceAcc               = dlarray(readmatrix(metricsFolder + "SurfaceAcc.csv"              ), 'BC');
SurfacePot               = dlarray(readmatrix(metricsFolder + "SurfacePot.csv"              ), 'BC');

% Load Network
net = load("src/training/net-SIREN.mat").net;

% Compute metrics
% ---------------------------------- | Preset function ----- | NN | Trajectory Data ------- | Acceleration Data ----- | Potential Data -------- |
[PlanesMetric               , PlanesRadius        ] = dlfeval(@presets.mpeLoss, net, PlanesTrj               , PlanesAcc               , PlanesPot               );
[GeneralizationMetric_0_1   , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, GeneralizationTrj_0_1   , GeneralizationAcc_0_1   , GeneralizationPot_0_1   );
[GeneralizationMetric_1_10  , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, GeneralizationTrj_1_10  , GeneralizationAcc_1_10  , GeneralizationPot_1_10  );
[GeneralizationMetric_10_100, GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, GeneralizationTrj_10_100, GeneralizationAcc_10_100, GeneralizationPot_10_100);
[GeneralizationMetric       , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, GeneralizationTrj       , GeneralizationAcc       , GeneralizationPot       );
[SurfaceMetric              , SurfaceRadius       ] = dlfeval(@presets.mpeLoss, net, SurfaceTrj              , SurfaceAcc              , SurfacePot              );

fprintf("\n### Mean Percent Error (MPE) ###\n");
fprintf("Planes metric                   : %f\n", mean(PlanesMetric               ));
fprintf("Generalization metric [0R:1R]   : %f\n", mean(GeneralizationMetric_0_1   ));
fprintf("Generalization metric [1R:10R]  : %f\n", mean(GeneralizationMetric_1_10  ));
fprintf("Generalization metric [10R:100R]: %f\n", mean(GeneralizationMetric_10_100));
fprintf("Generalization metric [0R:100R] : %f\n", mean(GeneralizationMetric       ));
fprintf("Surface metric                  : %f\n", mean(SurfaceMetric              ));

% Plotting
if headless
    return;
end

% Generalization: mpeLoss vs. distance(R), convert mpeLoss in log scale
figure;
plot(GeneralizationRadius, GeneralizationMetric, '.', 'DisplayName', 'Generalization');
set(gca, 'YScale', 'log');
xlabel('Distance (R)');
xlim([0, 20]);
ylabel('Mean Percent Error (MPE)');
title('Generalization: MPE vs. Distance (R)');
legend('show'); 
grid on;

% Planes: heatmap of 3 planes, value=mpeLoss(i), 2Dposition=PlanesTrj, planeID= which value of PlanesTrj(1:3,i) is 0
figure;
title('Planes: MPE');
XYi = 0 == PlanesTrj(3, :);
XZi = 0 == PlanesTrj(2, :);
YZi = 0 == PlanesTrj(1, :);
XY  = extractdata(PlanesTrj(1:2  , XYi));
XZ  = extractdata(PlanesTrj(1:2:3, XZi));
YZ  = extractdata(PlanesTrj(2:3  , YZi));
XYm = PlanesMetric(XYi);
XZm = PlanesMetric(XZi);
YZm = PlanesMetric(YZi);
subplot(1, 3, 1);
scatter(XY(1, :), XY(2, :), 2e2, XYm, '.');
title('XY Plane');
subplot(1, 3, 2);
scatter(XZ(1, :), XZ(2, :), 2e2, XZm, '.');
title('XZ Plane');
subplot(1, 3, 3);
scatter(YZ(1, :), YZ(2, :), 2e2, YZm, '.');
title('YZ Plane');
colormap jet;
colorbar;

% Surface: 3d plot of surface, with color value depending on the mpeLoss
figure;
scatter3(extractdata(SurfaceTrj(1, :)), extractdata(SurfaceTrj(2, :)), extractdata(SurfaceTrj(3, :)), 2e2, SurfaceMetric, 'o', 'filled');
title('Surface: MPE');