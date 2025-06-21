% This script tests the performance of the trained model.
%
% File: runTest.m
%     entrypoint for Testing

metricsFolder   = "src/preprocessing/datastore/metrics/";
headless        = batchStartupOptionUsed;

% Preparations - Data
data = mLoadData("src/preprocessing/metricsData.mat");
net  = load("src/training/residual.mat").net;

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

% Compute Polyhedral comparison
[PlanesPoly               , PlanesRadiusPoly        ] = presets.comparePolyhedral(data.mPlanesTRJ               , data.mPlanesACC               , data.mPlanesPOT               , data.pPlanesTRJ               , data.pPlanesACC               , data.pPlanesPOT               );
[GeneralizationPoly_0_1   , GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ_0_1   , data.mGeneralizationACC_0_1   , data.mGeneralizationPOT_0_1   , data.pGeneralizationTRJ_0_1   , data.pGeneralizationACC_0_1   , data.pGeneralizationPOT_0_1   );
[GeneralizationPoly_1_10  , GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ_1_10  , data.mGeneralizationACC_1_10  , data.mGeneralizationPOT_1_10  , data.pGeneralizationTRJ_1_10  , data.pGeneralizationACC_1_10  , data.pGeneralizationPOT_1_10  );
[GeneralizationPoly_10_100, GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ_10_100, data.mGeneralizationACC_10_100, data.mGeneralizationPOT_10_100, data.pGeneralizationTRJ_10_100, data.pGeneralizationACC_10_100, data.pGeneralizationPOT_10_100);
[GeneralizationPoly       , GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ       , data.mGeneralizationACC       , data.mGeneralizationPOT       , data.pGeneralizationTRJ       , data.pGeneralizationACC       , data.pGeneralizationPOT       );
[SurfacePoly              , SurfaceRadiusPoly       ] = presets.comparePolyhedral(data.mSurfaceTRJ              , data.mSurfaceACC              , data.mSurfacePOT              , data.pSurfaceTRJ              , data.pSurfaceACC              , data.pSurfacePOT              );


% Plotting
if headless
    return;
end

% Generalization: mpeLoss vs. distance(R), convert mpeLoss in log scale
plotGeneralization(GeneralizationRadius, GeneralizationMetric, GeneralizationRadiusPoly, GeneralizationPoly);

% TODO: Predict potentials within the network
%actNN               = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'scaleNNPotentialLayer');
%actLF               = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'analyticModelLayer'   );
%actFuse             = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'fuseModelsLayer'      );
%[sortedRadius, idx] = sort(extractdata(GeneralizationRadius));
%sortedNN            = abs(extractdata(actNN(idx)  ));
%sortedLF            = abs(extractdata(actLF(idx)  ));
%sortedFuse          = abs(extractdata(actFuse(idx)));

% TODO: Implement more visualizations
%figure;
%hold on;
%semilogy(sortedRadius, XXX , '.', 'DisplayName', 'XXX');

%set(gca, 'YScale', 'log');
%xlim([0, 20]);
%grid on;
%xline(10, '--', 'R = 10', 'LabelVerticalAlignment', 'bottom');
%xlabel('XXX');
%ylabel('XXX');
%title('XXX');
%legend('show');


% Planes Metric
plotPlanes(data.mPlanesTRJ, PlanesMetric);
plotPlanes(data.pPlanesTRJ, PlanesPoly);

% Surface Metric
plotSurface(data.mSurfaceTRJ, SurfaceMetric);
plotSurface(data.pSurfaceTRJ, SurfacePoly);

% clearvars
clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



function data = mLoadData(path)
    data = load(path);
    data.mGeneralizationTRJ = cat(1, data.mGeneralizationTRJ_0_1, data.mGeneralizationTRJ_1_10, data.mGeneralizationTRJ_10_100);
    data.mGeneralizationACC = cat(1, data.mGeneralizationACC_0_1, data.mGeneralizationACC_1_10, data.mGeneralizationACC_10_100);
    data.mGeneralizationPOT = cat(1, data.mGeneralizationPOT_0_1, data.mGeneralizationPOT_1_10, data.mGeneralizationPOT_10_100);
    data.pGeneralizationTRJ = cat(1, data.pGeneralizationTRJ_0_1, data.pGeneralizationTRJ_1_10, data.pGeneralizationTRJ_10_100);
    data.pGeneralizationACC = cat(1, data.pGeneralizationACC_0_1, data.pGeneralizationACC_1_10, data.pGeneralizationACC_10_100);
    data.pGeneralizationPOT = cat(1, data.pGeneralizationPOT_0_1, data.pGeneralizationPOT_1_10, data.pGeneralizationPOT_10_100);

    names = fieldnames(data);
    for i = 1:numel(names)
        data.(names{i}) = dlarray(data.(names{i}), 'BC');
    end
end