% PINN-GravityModel
% File: main.m
%     entrypoint for Data and Training
% Authors:
%     Andrea Valentinuzzi 2090451
%     Giovanni Brejc 2096046

close all; clear; clc;
addpath(genpath("src"));
DO_DATA_EXTRACTION = false;
DO_PREPROCESSING   = true;
DO_TRAINING        = true;
DO_TESTING         = true;

% Dataset
if true == DO_DATA_EXTRACTION
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting dataset extraction']);
    run("src/data/runData.m");
end

% Preprocessing
if true == DO_PREPROCESSING
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting preprocessing']);
    run("src/preprocessing/runPreprocessing.m");
end

% Training
if true == DO_TRAINING
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting training']);
    run("src/training/runTraining.m");
end

% Test
if true == DO_TESTING
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting testing']);
    run("src/test/runTest.m");
end