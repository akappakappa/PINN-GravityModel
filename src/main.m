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

% -------------------------------------------------------------------------

% PINN-GravityModel
% File: main.m
%     entrypoint for Data and Training
% Authors:
%     Andrea Valentinuzzi 2090451
%     Giovanni Brejc 2096046

close all; clear all; clc;
DEBUG = true;         % Extra debug information
NEWDATASET = false;   % Generates new dataset
EZMODE = false;       % Disables custom training loop
PLOTNET = false;      % Plots the network
GPU = "auto";         % GPU acceleration

% Dataset
if true == NEWDATASET
    run("src/data/runData.m")
else
    dataset = load("src/data/dataset.mat");
end

% Preprocessing
run("src/preprocessing/runPreprocessing.m");

% Training
run("src/training/runTraining.m");

% Test
run("src/test/runTest.m");