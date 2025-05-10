baseDir = "Trajectories/";

function data = loadData(path)
    data = double(py.pickle.load(py.open(path, "rb")));
end

function path = goDeep(path, match)
    oneDeeper = dir(path);
    oneDeeper = oneDeeper(contains({oneDeeper.name}, match)).name;
    path      = path + oneDeeper + "/";
end

% Test Surface data
folder      = "SurfaceDist/";
match       = ["eros", "HeterogeneousPoly"];
folder      = goDeep(baseDir + folder, match(1));
tSurfaceTRJ = loadData(folder + "trajectory.data");
folder      = goDeep(folder, match(2));
tSurfaceACC = loadData(folder + "acceleration.data");
tSurfacePOT = loadData(folder + "potential.data");

% Training Random data
folder     = "RandomDist/";
match      = ["N_90000", "HeterogeneousPoly"];
folder     = goDeep(baseDir + folder, match(1));
tRandomTRJ = loadData(folder + "trajectory.data");
folder     = goDeep(folder, match(2));
tRandomACC = loadData(folder + "acceleration.data");
tRandomPOT = loadData(folder + "potential.data");

% Metrics Planes data
folder     = "PlanesDist/";
match      = ["eros", "HeterogeneousPoly"];
folder     = goDeep(baseDir + folder, match(1));
mPlanesTRJ = loadData(folder + "trajectory.data");
folder     = goDeep(folder, match(2));
mPlanesACC = loadData(folder + "acceleration.data");
mPlanesPOT = loadData(folder + "potential.data");

% Metrics Random data
folder                    = "RandomDist/";
match_0_1                 = ["N_500_RadBounds[0.0, 16000.0]"       , "HeterogeneousPoly"];
match_1_10                = ["N_500_RadBounds[16000.0, 160000.0]"  , "HeterogeneousPoly"];
match_10_100              = ["N_500_RadBounds[160000.0, 1600000.0]", "HeterogeneousPoly"];
folder_0_1                = goDeep(baseDir + folder, match_0_1(1));
folder_1_10               = goDeep(baseDir + folder, match_1_10(1));
folder_10_100             = goDeep(baseDir + folder, match_10_100(1));
mGeneralizationTRJ_0_1    = loadData(folder_0_1    + "trajectory.data");
mGeneralizationTRJ_1_10   = loadData(folder_1_10   + "trajectory.data");
mGeneralizationTRJ_10_100 = loadData(folder_10_100 + "trajectory.data");
folder_0_1                = goDeep(folder_0_1   , match_0_1(2));
folder_1_10               = goDeep(folder_1_10  , match_1_10(2));
folder_10_100             = goDeep(folder_10_100, match_10_100(2));
mGeneralizationACC_0_1    = loadData(folder_0_1    + "acceleration.data");
mGeneralizationACC_1_10   = loadData(folder_1_10   + "acceleration.data");
mGeneralizationACC_10_100 = loadData(folder_10_100 + "acceleration.data");
mGeneralizationPOT_0_1    = loadData(folder_0_1    + "potential.data");
mGeneralizationPOT_1_10   = loadData(folder_1_10   + "potential.data");
mGeneralizationPOT_10_100 = loadData(folder_10_100 + "potential.data");

% Metrics Surface data
folder      = "SurfaceDHGridDist/";
match       = ["eros", "HeterogeneousPoly"];
folder      = goDeep(baseDir + folder, match(1));
mSurfaceTRJ = loadData(folder + "trajectory.data");
folder      = goDeep(folder, match(2));
mSurfaceACC = loadData(folder + "acceleration.data");
mSurfacePOT = loadData(folder + "potential.data");

% Transpose POTs
tSurfacePOT               = tSurfacePOT.';
tRandomPOT                = tRandomPOT.';
mPlanesPOT                = mPlanesPOT.';
mGeneralizationPOT_0_1    = mGeneralizationPOT_0_1.';
mGeneralizationPOT_1_10   = mGeneralizationPOT_1_10.';
mGeneralizationPOT_10_100 = mGeneralizationPOT_10_100.';
mSurfacePOT               = mSurfacePOT.';

% Combine Generalization data
mGeneralizationTRJ = cat(1, mGeneralizationTRJ_0_1, mGeneralizationTRJ_1_10, mGeneralizationTRJ_10_100);
mGeneralizationACC = cat(1, mGeneralizationACC_0_1, mGeneralizationACC_1_10, mGeneralizationACC_10_100);
mGeneralizationPOT = cat(1, mGeneralizationPOT_0_1, mGeneralizationPOT_1_10, mGeneralizationPOT_10_100);

% Saving
save("dataset.mat", ...
    "tSurfaceTRJ"       , "tSurfaceACC"       , "tSurfacePOT"       , ...
    "tRandomTRJ"        , "tRandomACC"        , "tRandomPOT"        , ...
    "mPlanesTRJ"        , "mPlanesACC"        , "mPlanesPOT"        , ...
    "mGeneralizationTRJ", "mGeneralizationACC", "mGeneralizationPOT", ...
    "mSurfaceTRJ"       , "mSurfaceACC"       , "mSurfacePOT"         ...
);
clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING