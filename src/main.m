% PINN-GravityModel
% File: main.m
%     entrypoint for Data and Training
% Authors:
%     Andrea Valentinuzzi 2090451
%     Giovanni Brejc 2096046

close all; clear all; clc;
addpath(genpath("src"));
SKIP_DATASET_GENERATION = true;
SKIP_PREPROCESSING = true;

% Dataset
if true == SKIP_DATASET_GENERATION
    dataset = load("src/data/dataset.mat");
else
    run("src/data/runData.m");
end

% Preprocessing
if true == SKIP_PREPROCESSING
else
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting preprocessing']);
    run("src/preprocessing/runPreprocessing.m");
end


% Training
disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting training']);
run("src/training/runTraining.m");

% Test
disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting test']);
run("src/test/runTest.m");