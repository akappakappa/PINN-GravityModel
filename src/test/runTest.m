% This script tests the performance of the trained model.
%
% File: runTest.m
%     entrypoint for Testing

metricsFolder   = "src/preprocessing/datastore/metrics/";
headless        = batchStartupOptionUsed;

% Preparations - Data
data = mLoadData("src/preprocessing/metricsData.mat");
net  = load("src/training/net.mat").net;

% Compute metrics
% ---------------------------------------------------------- | Preset func -- | NN | Trajectory Data ------------- | Acceleration Data ----------- | Potential Data -------------- |
[PlanesMetric               , PlanesRadius        ] = dlfeval(@presets.mpeLoss, net, data.mPlanesTRJ               , data.mPlanesACC               , data.mPlanesPOT               );
[GeneralizationMetric_0_1   , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_0_1   , data.mGeneralizationACC_0_1   , data.mGeneralizationPOT_0_1   );
[GeneralizationMetric_1_10  , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_1_10  , data.mGeneralizationACC_1_10  , data.mGeneralizationPOT_1_10  );
[GeneralizationMetric_10_100, GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_10_100, data.mGeneralizationACC_10_100, data.mGeneralizationPOT_10_100);
[GeneralizationMetric       , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ       , data.mGeneralizationACC       , data.mGeneralizationPOT       );
[SurfaceMetric              , SurfaceRadius       ] = dlfeval(@presets.mpeLoss, net, data.mSurfaceTRJ              , data.mSurfaceACC              , data.mSurfacePOT              );

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
semilogy(extractdata(GeneralizationRadius), extractdata(GeneralizationMetric), '.', 'DisplayName', 'Generalization');

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xlabel('Distance (R)');
ylabel('Mean Percent Error (MPE)');
title('Generalization: MPE vs. Distance (R)');
legend('show'); 

% Predicting potentials within the network
actNN               = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'scaleNNPotentialLayer');
actLF               = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'analyticModelLayer'   );
actFuse             = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'fuseModelsLayer'      );
[sortedRadius, idx] = sort(extractdata(GeneralizationRadius));
sortedNN            = abs(extractdata(actNN(idx)  ));
sortedLF            = abs(extractdata(actLF(idx)  ));
sortedFuse          = abs(extractdata(actFuse(idx)));

% Plotting potentials
figure;
hold on;
semilogy(sortedRadius, sortedLF  , '.', 'DisplayName', 'PotAnalytic');
semilogy(sortedRadius, sortedFuse, '.', 'DisplayName', 'PotFused'   );

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xline(10, '--', 'R = 10', 'LabelVerticalAlignment', 'bottom');
xlabel('Distance (R)');
ylabel('Potential');
title('Generalization Potential: Analytic vs Fused');
legend('show');

% NN vs Analytic (Fusion)
figure;
semilogy(sortedRadius, abs(sortedNN - sortedLF), '.', 'DisplayName', 'NN - Analytic');

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xline(10, '--', 'R = 10', 'LabelVerticalAlignment', 'bottom');
xlabel('Distance (R)');
ylabel('Absolute Difference');
title('Generalization Potential: Difference between NN and Analytic Potential');
legend('show');

% Fused vs Analytic (Boundary)
figure;
semilogy(sortedRadius, sortedNN, '.', 'DisplayName', 'NN (= Fused - Analytic)');

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xline(10, '--', 'R = 10', 'LabelVerticalAlignment', 'bottom');
xlabel('Distance (R)');
ylabel('Absolute Difference');
title('Generalization Potential: NN');
legend('show');

return;

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

clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



function data = mLoadData(path)
    data = load(path);
    data.mGeneralizationTRJ = cat(1, data.mGeneralizationTRJ_0_1, data.mGeneralizationTRJ_1_10, data.mGeneralizationTRJ_10_100);
    data.mGeneralizationACC = cat(1, data.mGeneralizationACC_0_1, data.mGeneralizationACC_1_10, data.mGeneralizationACC_10_100);
    data.mGeneralizationPOT = cat(1, data.mGeneralizationPOT_0_1, data.mGeneralizationPOT_1_10, data.mGeneralizationPOT_10_100);

    names = fieldnames(data);
    for i = 1:numel(names)
        data.(names{i}) = dlarray(data.(names{i}), 'BC');
    end
end