% This script extracts data from the specified folders and saves it in a dataset.mat file.
%
% File: runData.m
%     entrypoint for Data Extraction
%     Data is organized in trajectory.data, acceleration.data, and potential.data files.

baseDir   = "Trajectories/";
objPath   = "Model/eros_shape_200700.obj";

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
match                     = ["N_500", "HeterogeneousPoly"];
bounds                    = {[0, 1], [1, 10], [10, 100]};
for bound = bounds
    % List all folders containing data within bounds
    matches = listMatchesWithinBounds(baseDir + folder, match(1), bound{1});

    % Aggregate data within such bounds
    [tempTRJ, tempACC, tempPOT] = deal([]);
    for i = 1:length(matches)
        tempFolder = goDeep(baseDir + folder, matches(i));
        tempTRJ    = [tempTRJ; loadData(tempFolder + "trajectory.data")];
        tempFolder = goDeep(tempFolder, match(2));
        tempACC    = [tempACC; loadData(tempFolder + "acceleration.data")];
        tempPOT    = [tempPOT, loadData(tempFolder + "potential.data")];
    end

    % Save data
    switch bound{1}(1)
        case 0
            mGeneralizationTRJ_0_1 = tempTRJ;
            mGeneralizationACC_0_1 = tempACC;
            mGeneralizationPOT_0_1 = tempPOT;
        case 1
            mGeneralizationTRJ_1_10 = tempTRJ;
            mGeneralizationACC_1_10 = tempACC;
            mGeneralizationPOT_1_10 = tempPOT;
        case 10
            mGeneralizationTRJ_10_100 = tempTRJ;
            mGeneralizationACC_10_100 = tempACC;
            mGeneralizationPOT_10_100 = tempPOT;
        otherwise
            error("Invalid bounds");
    end
end

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

% Masking asteroid in Planes data
mesh   = py.trimesh.load_mesh(objPath);
rayObj = py.trimesh.ray.ray_triangle.RayMeshIntersector(mesh);
N      = size(mPlanesTRJ, 1);
step   = 100;
mask   = true(N, 1);
for i = 1:step:N
    endIdx = (floor(i / step) + 1) * step;
    planesSubset = mPlanesTRJ(i:endIdx, :) / 1e3;
    mask(i:endIdx) = ~logical(rayObj.contains_points(planesSubset));
end
mPlanesTRJ = mPlanesTRJ(mask, :);
mPlanesACC = mPlanesACC(mask, :);
mPlanesPOT = mPlanesPOT(mask, :);

% Saving
save("dataset.mat", ...
    "tSurfaceTRJ"              , "tSurfaceACC"              , "tSurfacePOT"              , ...
    "tRandomTRJ"               , "tRandomACC"               , "tRandomPOT"               , ...
    "mPlanesTRJ"               , "mPlanesACC"               , "mPlanesPOT"               , ...
    "mGeneralizationTRJ_0_1"   , "mGeneralizationACC_0_1"   , "mGeneralizationPOT_0_1"   , ...
    "mGeneralizationTRJ_1_10"  , "mGeneralizationACC_1_10"  , "mGeneralizationPOT_1_10"  , ...
    "mGeneralizationTRJ_10_100", "mGeneralizationACC_10_100", "mGeneralizationPOT_10_100", ...
    "mSurfaceTRJ"              , "mSurfaceACC"              , "mSurfacePOT"                ...
);
clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



function data = loadData(path)
    % LOADDATA  Load Python Pickle data from a file.
    %   DATA = LOADDATA(PATH) loads the data from the file specified by PATH, casting it to double.

    data = double(py.pickle.load(py.open(path, "rb")));
end

function path = goDeep(path, match)
    % GODEEP  Travel one level deeper in the directory structure.
    %   PATH = GODEEP(PATH, MATCH) travels one level deeper in the directory structure, looking for a MATCH in the directory name.

    oneDeeper = dir(path);
    oneDeeper = oneDeeper(contains({oneDeeper.name}, match)).name;
    path      = path + oneDeeper + "/";
end

function matches = listMatchesWithinBounds(path, match, bound)
    % LISTMATCHESWITHINBOUNDS  List all dataset folders with data within specified bounds.
    %   MATCHES = LISTMATCHESWITHINBOUNDS(PATH, MATCH, BOUND) lists all dataset folders matching the MATCH string, with data within the specified BOUNDS.

    list = [];
    for i = bound(1):bound(2) - 1
        list = [list, match + "_RadBounds[" + 16000 * i + ".0"];
    end
    matches   = dir(path);
    matches   = {matches(contains({matches.name}, list)).name};
end