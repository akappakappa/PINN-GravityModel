% This script tests the performance of the trained model.
%
%   Plotting 5 metrics
%       - GENERALIZATION interior
%       - GENERALIZATION exterior
%       - GENERALIZATION extrapolation
%       - SURFACE
%       - PLANES
%
%   ..and comparing them to the Polyhedral model

metricsFolder   = "src/preprocessing/datastore/metrics/";
headless        = batchStartupOptionUsed;

% Preparations - Data
data  = mLoadData("src/preprocessing/metricsData.mat");
nname = "net";
net   = load("src/training/" + nname + ".mat").net;

% Compute metrics
% ---------------------------------------------------------- | Preset func -- | NN | Trajectory Data ------------- | Acceleration Data ----------- | Potential Data -------------- |
[PlanesMetric               , PlanesRadius        ] = dlfeval(@presets.mpeLoss, net, data.mPlanesTRJ               , data.mPlanesACC               , data.mPlanesPOT               );
[GeneralizationMetric_0_1   , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_0_1   , data.mGeneralizationACC_0_1   , data.mGeneralizationPOT_0_1   );
[GeneralizationMetric_1_10  , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_1_10  , data.mGeneralizationACC_1_10  , data.mGeneralizationPOT_1_10  );
[GeneralizationMetric_10_100, GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_10_100, data.mGeneralizationACC_10_100, data.mGeneralizationPOT_10_100);
[GeneralizationMetric       , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ       , data.mGeneralizationACC       , data.mGeneralizationPOT       );
[SurfaceMetric              , SurfaceRadius       ] = dlfeval(@presets.mpeLoss, net, data.mSurfaceTRJ              , data.mSurfaceACC              , data.mSurfacePOT              );

fprintf("\n### Mean Percent Error (MPE) ###\n");
fprintf("GNR TOT          : %f\n", mean(GeneralizationMetric       ));
fprintf("GNR INTERIOR     : %f\n", mean(GeneralizationMetric_0_1   ));
fprintf("GNR EXTERIOR     : %f\n", mean(GeneralizationMetric_1_10  ));
fprintf("GNR EXTRAPOLATION: %f\n", mean(GeneralizationMetric_10_100));
fprintf("SRF              : %f\n", mean(SurfaceMetric              ));
fprintf("PLN              : %f\n", mean(PlanesMetric               ));

% Compute Polyhedral comparison
[PlanesPoly               , PlanesRadiusPoly        ] = presets.comparePolyhedral(data.mPlanesTRJ               , data.mPlanesACC               , data.mPlanesPOT               , data.pPlanesTRJ               , data.pPlanesACC               , data.pPlanesPOT               );
[GeneralizationPoly_0_1   , GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ_0_1   , data.mGeneralizationACC_0_1   , data.mGeneralizationPOT_0_1   , data.pGeneralizationTRJ_0_1   , data.pGeneralizationACC_0_1   , data.pGeneralizationPOT_0_1   );
[GeneralizationPoly_1_10  , GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ_1_10  , data.mGeneralizationACC_1_10  , data.mGeneralizationPOT_1_10  , data.pGeneralizationTRJ_1_10  , data.pGeneralizationACC_1_10  , data.pGeneralizationPOT_1_10  );
[GeneralizationPoly_10_100, GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ_10_100, data.mGeneralizationACC_10_100, data.mGeneralizationPOT_10_100, data.pGeneralizationTRJ_10_100, data.pGeneralizationACC_10_100, data.pGeneralizationPOT_10_100);
[GeneralizationPoly       , GeneralizationRadiusPoly] = presets.comparePolyhedral(data.mGeneralizationTRJ       , data.mGeneralizationACC       , data.mGeneralizationPOT       , data.pGeneralizationTRJ       , data.pGeneralizationACC       , data.pGeneralizationPOT       );
[SurfacePoly              , SurfaceRadiusPoly       ] = presets.comparePolyhedral(data.mSurfaceTRJ              , data.mSurfaceACC              , data.mSurfacePOT              , data.pSurfaceTRJ              , data.pSurfaceACC              , data.pSurfacePOT              );


% STOP in batch mode
if headless
    return;
end


% Single values
if ~exist("../../fig", 'dir')
    mkdir("../../fig");
end
if ~exist(fullfile("../../fig", nname), 'dir')
    mkdir(fullfile("../../fig", nname));
end

labels = {'GNR TOT'; 'GNR INTERIOR'; 'GNR EXTERIOR'; 'GNR EXTRAPOLATION'; 'SRF'; 'PLN'};
values = [mean(GeneralizationMetric); mean(GeneralizationMetric_0_1); mean(GeneralizationMetric_1_10); mean(GeneralizationMetric_10_100); mean(SurfaceMetric); mean(PlanesMetric)];
T = table(labels, values);
T.Properties.VariableNames = {'Metric', 'Value'};
writetable(T, "../../fig/" + nname + "/MTR.csv");

% Generalization: mpeLoss vs. distance(R), convert mpeLoss in log scale
plotGeneralization(nname, GeneralizationRadius, GeneralizationMetric, GeneralizationRadiusPoly, GeneralizationPoly);

% Planes Metric
plotPlanes(nname, true, data.mPlanesTRJ, PlanesMetric);
plotPlanes(nname, false, data.pPlanesTRJ, PlanesPoly);

% Surface Metric
plotSurface(nname, true, data.mSurfaceTRJ, SurfaceMetric);
plotSurface(nname, false, data.pSurfaceTRJ, SurfacePoly);


clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



% --- HELPER FUNCTIONS ---
function data = mLoadData(path)
    % Load metrics dataset

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