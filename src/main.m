% Entrypoint for the project
% This script simply manages the execution of the other scripts, depending on DO_* flags
%
%   Authors: Andrea Valentinuzzi, Giovanni Brejc

close all; clear; clc;
addpath(genpath(fileparts(mfilename('fullpath'))));
if batchStartupOptionUsed
    % Run batch mode as
    %   cd path/to/PINN-GravityModel
    %   srun matlab -batch "run('src/main.m')"
    cd ..
end

DO_DATA_EXTRACTION   = false;   % Extract data from python pickle-formatted dataset of pos,acc,pot values
DO_PREPROCESSING     = false;   % Perform dataset-wide preprocessing
DO_TRAINING          = true;    % Run training and save resulting network
DO_TESTING           = true;    % Test the model and save relevant figures

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