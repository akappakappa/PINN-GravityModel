% --- Cost
% J(Θ) = 1/Nf*[∑(i=1,Nf)|ai+∇Û(ri|Θ)|^2]

% --- Network
% N {10,20,40,80} per layer, 8 hidden layers
% 10 -> too small .. 80 -> overfitting .. 40 -> best (empirical)

% --- Hyperparameters
% BatchSize = 262144 -> for little available VRAM
% LearningRate η0 = 0.005 -> decay after epoch i>=i0 : ηi = η0*pow(α,-(i-i0)/σ)
% ReferenceEpoch i0 = 25000, ScaleFactor σ = 25000, DecayRate α = 0.5
% ActivationFunction = GELU
% Epochs = 100000
% Optimizer = Adam
% Initializer = Glorot uniform
% x Transform (preprocessing to fit r) = MinMax along each component
% y Transform (preprocessing to fit a) = uniform MinMax across all components

% PINN-GM-III
%   INPUT: x,y,z
%   FEATURE ENGINEERING:
%   NEURAL NETWORK

% open data/eros.pk file, which represents a mascon model of Eros asteroid
%   the original python script suggests:
%   with open("Itokawa_mascon.pk", "rb") as file:
%       points, masses, name = pk.load(file)
%   translate now to matlab

% -------------------------------------------------------------------------

% PINN-GravityModel
% File: main.m
%     entrypoint for Dataset choice, Parameters choice and Training
% Authors:
%     Andrea Valentinuzzi 2090451
%     Giovanni Brejc XXXXXXX

% Initialize MATLAB
close all; clear all; clc;
LOG = input("Display LOGs? (y/N)", "s");
switch LOG
    case {"y", "Y"}
        LOG = true;
    case {"n", "N", []}
        LOG = false;
    otherwise
        disp("[WARN] Assuming 'NO'");
        LOG = false;
end

% DATASET - Choose
choiceDataset = input("Choose the dataset on which to perform training:" + ...
    "\n1. Earth" + ...
    "\n2. Moon" + ...
    "\n3. Eros (default)", ...
    "s");
switch choiceDataset
    case {"1", "Earth", "earth"}
        choiceDataset = "src/data/earth.pk";
    case {"2", "Moon", "moon"}
        choiceDataset = "src/data/moon.pk";
    case {"3", "Eros", "eros", []}
        choiceDataset = "src/data/eros.pk";
    otherwise
        disp("[WARN] Input changed by default to 'Eros'");
        choiceDataset = "src/data/eros.pk";
end
if LOG, disp("[INFO] Dataset: " + choiceDataset), end
if ~exist(choiceDataset, "file")
    error("[ERROR] Dataset file " + choiceDataset + " NOT present, exiting..");
end
if LOG, disp(newline), end

% DATASET - Load
pyFile = py.open(choiceDataset, "rb");
pyData = py.pickle.load(pyFile);
masconPoints = double(pyData{1})';
if LOG, disp("[INFO] Received 'masconPoints' of size: " + string(mat2str(size(masconPoints))) + ", type: " + class(masconPoints)), end
masconMasses = double(pyData{2});
if LOG, disp("[INFO] Received 'masconMasses' of size: " + string(mat2str(size(masconMasses))) + ", type: " + class(masconMasses)), end
masconName = string(pyData{3});
if LOG, disp("[INFO] Loaded mascons for " + masconName + "!" + newline), end
clear pyFile pyData;