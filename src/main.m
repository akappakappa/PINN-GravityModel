% PINN-GravityModel
% File: main.m
%     entrypoint for Data and Training
% Authors:
%     Andrea Valentinuzzi 2090451
%     Giovanni Brejc 2096046

close all; clear all; clc;
addpath(genpath("src"));
SKIP_DATASET_GENERATION = true;

% Dataset
if false == SKIP_DATASET_GENERATION
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