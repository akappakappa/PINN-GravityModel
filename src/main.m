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
%     entrypoint for Data and Training
% Authors:
%     Andrea Valentinuzzi 2090451
%     Giovanni Brejc 2096046

close all; clear all; clc;
DEBUG = true;

% Options for the dataset
optDataset = input("Generate new dataset? (y/N)", "s");
switch optDataset
    case {"y", "Y"}
        disp("Generating new dataset...");
        run("src/data/runData.m");
    case {"n", "N", []}
        dataset = load("src/data/dataset.mat");
    otherwise
        error("Invalid option");
end

% Run training in training/run.m
run("src/training/runTraining.m");