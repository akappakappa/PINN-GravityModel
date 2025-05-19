% Physics-Informed Neural Network for Gravitational Potential estimation on 433 Eros.
%
% File: main.m
%     entrypoint for the project
%     This script simply manages the execution of the other scripts, depending on DO_* flags
% Authors:
%     Andrea     Valentinuzzi   2090451
%     Giovanni   Brejc          2096046

close all; clear; clc;
addpath(genpath(fileparts(mfilename('fullpath'))));
if batchStartupOptionUsed
    cd ..
end

DO_DATA_EXTRACTION   = false;
DO_PREPROCESSING     = false;
DO_TRAINING          = false;
DO_TESTING           = true;

if true == DO_DATA_EXTRACTION
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting dataset extraction']);
    run("src/data/runData.m");
end
if true == DO_PREPROCESSING
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting preprocessing'     ]);
    run("src/preprocessing/runPreprocessing.m");
end
if true == DO_TRAINING
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting training'          ]);
    run("src/training/runTraining.m");
end
if true == DO_TESTING
    disp(['[', char(datetime, 'dd MMM hh:mm'), '] [LOG] Starting testing'           ]);
    run("src/test/runTest.m");
end