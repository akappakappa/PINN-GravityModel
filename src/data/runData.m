% This script extracts data from the specified folders and saves it in a dataset.mat file.
%
%   Data is organized in trajectory.data, acceleration.data, and potential.data files.

baseDir   = "Trajectories/";
objPath   = "Model/eros_shape_200700.obj";

%{
    TRAINING Data
    - HeterogeneousPoly Surface
    - HeterogeneousPoly Random [0:10]
%}
% Training Surface data
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

%{
    METRICS Data
    - (true labels) HeterogeneousPoly Planes
    - (true labels) HeterogeneousPoly Random [0:100] = [0:1] + [1:10] + [10:100]
    - (true labels) HeterogeneousPoly Surface
    - (comparative) Polyhedral Planes
    - (comparative) Polyhedral Random [0:100] = [0:1] + [1:10] + [10:100]
    - (comparative) Polyhedral Surface
%}
% Metrics (true labels) HeterogeneousPoly Planes
folder     = "PlanesDist/";
match      = ["eros", "HeterogeneousPoly"];
folder     = goDeep(baseDir + folder, match(1));
mPlanesTRJ = loadData(folder + "trajectory.data");
folder     = goDeep(folder, match(2));
mPlanesACC = loadData(folder + "acceleration.data");
mPlanesPOT = loadData(folder + "potential.data");

% Metrics (comparative) Polyhedral Planes
folder     = "PlanesDist/";
match      = ["eros", "Polyhedral"];
folder     = goDeep(baseDir + folder, match(1));
pPlanesTRJ = loadData(folder + "trajectory.data");
folder     = goDeep(folder, match(2));
pPlanesACC = loadData(folder + "acceleration.data");
pPlanesPOT = loadData(folder + "potential.data");

% Metrics (true labels) HeterogeneousPoly Random [0:100] = [0:1] + [1:10] + [10:100]
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

% Metrics (comparative) Polyhedral Random [0:100] = [0:1] + [1:10] + [10:100]
folder                    = "RandomDist/";
match                     = ["N_500", "Polyhedral"];
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
            pGeneralizationTRJ_0_1 = tempTRJ;
            pGeneralizationACC_0_1 = tempACC;
            pGeneralizationPOT_0_1 = tempPOT;
        case 1
            pGeneralizationTRJ_1_10 = tempTRJ;
            pGeneralizationACC_1_10 = tempACC;
            pGeneralizationPOT_1_10 = tempPOT;
        case 10
            pGeneralizationTRJ_10_100 = tempTRJ;
            pGeneralizationACC_10_100 = tempACC;
            pGeneralizationPOT_10_100 = tempPOT;
        otherwise
            error("Invalid bounds");
    end
end

% Metrics (true labels) HeterogeneousPoly Surface
folder      = "SurfaceDist/";
match       = ["eros", "HeterogeneousPoly"];
folder      = goDeep(baseDir + folder, match(1));
mSurfaceTRJ = loadData(folder + "trajectory.data");
folder      = goDeep(folder, match(2));
mSurfaceACC = loadData(folder + "acceleration.data");
mSurfacePOT = loadData(folder + "potential.data");

% Metrics (comparative) Polyhedral Surface
folder      = "SurfaceDist/";
match       = ["eros", "Polyhedral"];
folder      = goDeep(baseDir + folder, match(1));
pSurfaceTRJ = loadData(folder + "trajectory.data");
folder      = goDeep(folder, match(2));
pSurfaceACC = loadData(folder + "acceleration.data");
pSurfacePOT = loadData(folder + "potential.data");

% Transpose POTs
tSurfacePOT               = tSurfacePOT.';
tRandomPOT                = tRandomPOT.';
mPlanesPOT                = mPlanesPOT.';
mGeneralizationPOT_0_1    = mGeneralizationPOT_0_1.';
mGeneralizationPOT_1_10   = mGeneralizationPOT_1_10.';
mGeneralizationPOT_10_100 = mGeneralizationPOT_10_100.';
mSurfacePOT               = mSurfacePOT.';
pPlanesPOT                = pPlanesPOT.';
pGeneralizationPOT_0_1    = pGeneralizationPOT_0_1.';
pGeneralizationPOT_1_10   = pGeneralizationPOT_1_10.';
pGeneralizationPOT_10_100 = pGeneralizationPOT_10_100.';
pSurfacePOT               = pSurfacePOT.';

% Masking asteroid in Planes data (true labels)
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

% Masking asteroid in Planes data (comparative)
mesh   = py.trimesh.load_mesh(objPath);
rayObj = py.trimesh.ray.ray_triangle.RayMeshIntersector(mesh);
N      = size(pPlanesTRJ, 1);
step   = 100;
mask   = true(N, 1);
for i = 1:step:N
    endIdx = (floor(i / step) + 1) * step;
    planesSubset = pPlanesTRJ(i:endIdx, :) / 1e3;
    mask(i:endIdx) = ~logical(rayObj.contains_points(planesSubset));
end
pPlanesTRJ = pPlanesTRJ(mask, :);
pPlanesACC = pPlanesACC(mask, :);
pPlanesPOT = pPlanesPOT(mask, :);

% Saving
save("dataset.mat", ...
    "tSurfaceTRJ"              , "tSurfaceACC"              , "tSurfacePOT"              , ...
    "tRandomTRJ"               , "tRandomACC"               , "tRandomPOT"               , ...
    "mPlanesTRJ"               , "mPlanesACC"               , "mPlanesPOT"               , ...
    "mGeneralizationTRJ_0_1"   , "mGeneralizationACC_0_1"   , "mGeneralizationPOT_0_1"   , ...
    "mGeneralizationTRJ_1_10"  , "mGeneralizationACC_1_10"  , "mGeneralizationPOT_1_10"  , ...
    "mGeneralizationTRJ_10_100", "mGeneralizationACC_10_100", "mGeneralizationPOT_10_100", ...
    "mSurfaceTRJ"              , "mSurfaceACC"              , "mSurfacePOT"              , ...
    "pPlanesTRJ"               , "pPlanesACC"               , "pPlanesPOT"               , ...
    "pGeneralizationTRJ_0_1"   , "pGeneralizationACC_0_1"   , "pGeneralizationPOT_0_1"   , ...
    "pGeneralizationTRJ_1_10"  , "pGeneralizationACC_1_10"  , "pGeneralizationPOT_1_10"  , ...
    "pGeneralizationTRJ_10_100", "pGeneralizationACC_10_100", "pGeneralizationPOT_10_100", ...
    "pSurfaceTRJ"              , "pSurfaceACC"              , "pSurfacePOT"                ...
);
clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



% --- HELPER FUNCTIONS ---
function data = loadData(path)
    % Load Python Pickle data from the file specified by path.

    data = double(py.pickle.load(py.open(path, "rb")));
end
function path = goDeep(path, match)
    % Travel one level deeper in the directory structure by matching subfolder name.

    oneDeeper = dir(path);
    oneDeeper = oneDeeper(contains({oneDeeper.name}, match)).name;
    path      = path + oneDeeper + "/";
end
function matches = listMatchesWithinBounds(path, match, bound)
    % List all dataset folders with data within specified bounds by matching subfolder name and bounds.

    list = [];
    for i = bound(1):bound(2) - 1
        list = [list, match + "_RadBounds[" + 16000 * i + ".0"];
    end
    matches   = dir(path);
    matches   = {matches(contains({matches.name}, list)).name};
end