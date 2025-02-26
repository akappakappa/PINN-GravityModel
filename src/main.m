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
NEWDATASET = false;   % Generates new dataset
GPU = "auto";         % GPU acceleration

% Dataset
if true == NEWDATASET
    run("src/data/runData.m")
else
    dataset = load("src/data/dataset.mat");
end

% Preprocessing
disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting preprocessing']);
run("src/preprocessing/runPreprocessing.m");

% Training
disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting training']);
run("src/training/runTraining.m");

% Test
disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting test']);
run("src/test/runTest.m");