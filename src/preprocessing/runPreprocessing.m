dataset = load("src/data/dataset.mat");

% Parameters
dataset.params  = struct();
splitPercentage = 0.99;
assert(splitPercentage > 0, 'split percentage must be greater than 0');
assert(splitPercentage < 1, 'split percentage must be less than 1');
dataset.params.splitPercentage = splitPercentage;

% Call preprocessing function preset
preprocessed = presets.PINN_GM_III(dataset);

% Create directories if they do not exist
if ~exist("datastore", "dir")
    mkdir("datastore");
end
if ~exist("datastore/train", "dir")
    mkdir("datastore/train");
end
if ~exist("datastore/validation", "dir")
    mkdir("datastore/validation");
end
if ~exist("datastore/metrics", "dir")
    mkdir("datastore/metrics");
end

% Save preprocessed training data
writematrix(preprocessed.trainTRJ     , "datastore/train/Trj.csv"     );
writematrix(preprocessed.trainACC     , "datastore/train/Acc.csv"     );
writematrix(preprocessed.trainPOT     , "datastore/train/Pot.csv"     );
writematrix(preprocessed.validationTRJ, "datastore/validation/Trj.csv");
writematrix(preprocessed.validationACC, "datastore/validation/Acc.csv");
writematrix(preprocessed.validationPOT, "datastore/validation/Pot.csv");

writestruct(preprocessed.params, "datastore/params.json");

% Save preprocessed metrics data
writematrix(preprocessed.mPlanesTRJ        , "datastore/metrics/PlanesTrj.csv");
writematrix(preprocessed.mPlanesACC        , "datastore/metrics/PlanesAcc.csv");
writematrix(preprocessed.mPlanesPOT        , "datastore/metrics/PlanesPot.csv");
writematrix(preprocessed.mGeneralizationTRJ, "datastore/metrics/GeneralizationTrj.csv");
writematrix(preprocessed.mGeneralizationACC, "datastore/metrics/GeneralizationAcc.csv");
writematrix(preprocessed.mGeneralizationPOT, "datastore/metrics/GeneralizationPot.csv");
writematrix(preprocessed.mSurfaceTRJ       , "datastore/metrics/SurfaceTrj.csv");
writematrix(preprocessed.mSurfaceACC       , "datastore/metrics/SurfaceAcc.csv");
writematrix(preprocessed.mSurfacePOT       , "datastore/metrics/SurfacePot.csv");

% Add folder to path recursively
addpath(genpath("datastore"));

clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING