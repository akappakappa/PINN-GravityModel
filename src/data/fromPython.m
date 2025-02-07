close all; clear all; clc;

% Python
py.importlib.import_module('numpy')

baseDir = "src/data/";

% Pickle
pickleSurfaceACC = py.pickle.load(py.open(baseDir + "SurfaceDist/erosN_200700/HeterogeneousPoly_eros_shape_200700_45014.18623266764_[5333.333333333333, 0, 0]_-45014.18623266764_[-5333.333333333333, 0, 0]_/acceleration.data", "rb"));
pickleSurfacePOT = py.pickle.load(py.open(baseDir + "SurfaceDist/erosN_200700/HeterogeneousPoly_eros_shape_200700_45014.18623266764_[5333.333333333333, 0, 0]_-45014.18623266764_[-5333.333333333333, 0, 0]_/potential.data", "rb"));
pickleSurfaceTRJ = py.pickle.load(py.open(baseDir + "SurfaceDist/erosN_200700/trajectory.data", "rb"));
pickleRandomdACC = py.pickle.load(py.open(baseDir + "RandomDist/eros_eros_shape_200700_N_90000_RadBounds[0.0, 160000.0]_UVol_True/HeterogeneousPoly_eros_shape_200700_45014.18623266764_[5333.333333333333, 0, 0]_-45014.18623266764_[-5333.333333333333, 0, 0]_/acceleration.data", "rb"));
pickleRandomdPOT = py.pickle.load(py.open(baseDir + "RandomDist/eros_eros_shape_200700_N_90000_RadBounds[0.0, 160000.0]_UVol_True/HeterogeneousPoly_eros_shape_200700_45014.18623266764_[5333.333333333333, 0, 0]_-45014.18623266764_[-5333.333333333333, 0, 0]_/potential.data", "rb"));
pickleRandomdTRJ = py.pickle.load(py.open(baseDir + "RandomDist/eros_eros_shape_200700_N_90000_RadBounds[0.0, 160000.0]_UVol_True/trajectory.data", "rb"));

% Txt
py.numpy.savetxt(baseDir + "surfaceACC.txt", pickleSurfaceACC);
py.numpy.savetxt(baseDir + "surfacePOT.txt", pickleSurfacePOT);
py.numpy.savetxt(baseDir + "surfaceTRJ.txt", pickleSurfaceTRJ);
py.numpy.savetxt(baseDir + "randomdACC.txt", pickleRandomdACC);
py.numpy.savetxt(baseDir + "randomdPOT.txt", pickleRandomdPOT);
py.numpy.savetxt(baseDir + "randomdTRJ.txt", pickleRandomdTRJ);

% Mat
surfaceACC = readmatrix(baseDir + "surfaceACC.txt");
surfacePOT = readmatrix(baseDir + "surfacePOT.txt");
surfaceTRJ = readmatrix(baseDir + "surfaceTRJ.txt");
randomdACC = readmatrix(baseDir + "randomdACC.txt");
randomdPOT = readmatrix(baseDir + "randomdPOT.txt");
randomdTRJ = readmatrix(baseDir + "randomdTRJ.txt");

% Save
save(baseDir + "dataset.mat", surfaceACC, surfacePOT, surfaceTRJ, randomdACC, randomdPOT, randomdTRJ);