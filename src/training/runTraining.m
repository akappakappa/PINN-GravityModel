% This script contains the training loop for the Physics-Informed NN model.
%
% File: runTraining.m
%     entrypoint for Training

executionEnvironment    = "auto";   % Leave as "auto" for GPU training if found, or set to "gpu" or "cpu"
headless                = batchStartupOptionUsed;
recoverFromCheckpoint   = false;    % Only set to true if you want to recover from a checkpoint on stopped training
useGPU                  = ("auto" == executionEnvironment || "gpu" == executionEnvironment) && canUseGPU;

% Preparations - Data
data         = tLoadDatastore("src/preprocessing/datastore");
net          = initialize(presets.network.PINN_GM_III(data.params.mu, data.params.e));
modelLoss    = dlaccelerate(@presets.loss.PINN_GM_III);
options      = presets.options.PINN_GM_III(data.params.split(1));
if useGPU
    net      = dlupdate(@gpuArray, net);
end
[tMBQ, vMBQ] = tSetupMinibatchQueues(data, options, executionEnvironment);

% Preparations - Loop
epoch         = 0;     % Epoch counter
iteration     = 0;     % Iteration counter
averageGrad   = [];    % Adam parameter
averageSqGrad = [];    % Adam parameter
bestNet       = net;   % Best network so far

% IF recovering from checkpoint on stopped training
if recoverFromCheckpoint
    checkpoint    = load("checkpoint");
    net           = checkpoint.bestNet;
    bestNet       = net;
    options       = checkpoint.options;    
    epoch         = checkpoint.epoch;
    averageGrad   = checkpoint.averageGrad;
    averageSqGrad = checkpoint.averageSqGrad;
end

% Preparations - Monitoring
monitor = tMakeMonitor(headless);
if options.verbose
    fprintf("|========================================================================================|\n");
    fprintf("|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Base Learning  |\n");
    fprintf("|         |             |   (hh:mm:ss)   |     Loss     |     Loss     |      Rate       |\n");
    fprintf("|========================================================================================|\n");
end

% Loop
start = tic;
while epoch < options.numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    shuffle(tMBQ);
    shuffle(vMBQ);

    % Loop over mini-batches
    while hasdata(tMBQ) && ~monitor.Stop
        iteration                         = iteration + 1;
        [Trj, Acc, Pot]                   = next(tMBQ);
        [loss, gradients, net.State]      = dlfeval(modelLoss, net, Trj, Acc, Pot, true);
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, options.learnRate);
        
        if ~headless
            recordMetrics(monitor, iteration, TrainingLoss = loss);
        end
        
        % Validation
        if (1 == iteration || 0 == mod(iteration, options.numIterationsPerEpoch)) && hasdata(vMBQ) && ~monitor.Stop
            [Trj, Acc, Pot]        = next(vMBQ);
            [validationLoss, ~, ~] = dlfeval(modelLoss, net, Trj, Acc, Pot, false);
            
            if ~headless
                recordMetrics(monitor, iteration, ValidationLoss = validationLoss);
            end
            if options.verbose
                D = duration(0, 0, toc(start), Format = "hh:mm:ss");
                fprintf("| %7d | %11d | %14s | %12.4f | %12.4f | %15.4f |\n"    , ...
                    epoch, iteration, D, loss, validationLoss, options.learnRate  ...
                );
            end

            % Snapshot
            if validationLoss < options.learnRateSchedule.BestValidationLoss
                bestNet = net;
            end

            % LR Schedule
            [options.learnRateSchedule, options.learnRate] = options.learnRateSchedule.update(options.learnRate, validationLoss);
        end

        % Monitoring
        if ~headless
            updateInfo(monitor, ...
                Epoch        = string(epoch) + " / " + string(options.numEpochs), ...
                Iteration    = iteration                                        , ...
                LearningRate = options.learnRate                                  ...
            );
            monitor.Progress = 100 * iteration / options.numIterations;
        end
    end

    % Checkpoints
    if 0 == mod(epoch, 2^6)
        save("checkpoint", "bestNet", "options", "epoch", "averageGrad", "averageSqGrad");
    end
end
if options.verbose
    fprintf("|========================================================================================|\n");
end

% Save
net = bestNet;
save("net", "net");

clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



function data = tLoadDatastore(path)
    % TLOADDATASTORE  Load the data from the specified path.
    %   DATA = TLOADDATASTORE(PATH) loads the data from the specified path, shuffling the training and validation sets.

    data            = struct();
    data.params     = readstruct(path + "/params.json");

    trainTrj        = readmatrix(path + "/train/Trj.csv");
    trainAcc        = readmatrix(path + "/train/Acc.csv");
    trainPot        = readmatrix(path + "/train/Pot.csv");
    data.train      = shuffle(combine( ...
        arrayDatastore(trainTrj)     , ...
        arrayDatastore(trainAcc)     , ...
        arrayDatastore(trainPot)       ...
    ));

    validationTrj   = readmatrix(path + "/validation/Trj.csv");
    validationAcc   = readmatrix(path + "/validation/Acc.csv");
    validationPot   = readmatrix(path + "/validation/Pot.csv");
    data.validation = shuffle(combine( ...
        arrayDatastore(validationTrj), ...
        arrayDatastore(validationAcc), ...
        arrayDatastore(validationPot)  ...
    ));
end

function [tMBQ, vMBQ] = tSetupMinibatchQueues(data, options, executionEnvironment)
    % TSETUPMINIBATCHQUEUES  Setup the mini-batch queues for training and validation.
    %   [TMBQ, VMBQ] = TSETUPMINIBATCHQUEUES(DATA, OPTIONS, EXECUTIONENVIRONMENT) sets up the mini-batch queues for training and validation, given OPTIONS and EXECUTIONENVIRONMENT.
    
    tMBQ = minibatchqueue(data.train                                           , ...
        MiniBatchFcn      = @(Trj, Acc, Pot) preprocessMiniBatch(Trj, Acc, Pot), ...
        MiniBatchSize     = options.miniBatchSize                              , ...
        OutputEnvironment = executionEnvironment                               , ...
        MiniBatchFormat   = 'BC'                                                 ...
    );
    
    vMBQ = minibatchqueue(data.validation                                                                              , ...
        MiniBatchFcn      = @(Trj, Acc, Pot) preprocessMiniBatch(Trj, Acc, Pot)                                        , ...
        MiniBatchSize     = floor(data.params.split(2) / options.numIterationsPerEpoch) * options.numIterationsPerEpoch, ...
        OutputEnvironment = executionEnvironment                                                                       , ...
        MiniBatchFormat   = 'BC'                                                                                         ...
    );



    function [Trj, Acc, Pot] = preprocessMiniBatch(Trj, Acc, Pot)
        % PREPROCESSMINIBATCH  Preprocess the mini-batch data.
        %   [TRJ, ACC, POT] = PREPROCESSMINIBATCH(TRJ, ACC, POT) preprocesses the mini-batch data.

        Trj = cat(1, Trj{:});
        Acc = cat(1, Acc{:});
        Pot = cat(1, Pot{:});
    end
end

function monitor = tMakeMonitor(headless)
    % TMAKEMONITOR  Create a training progress monitor.
    %   MONITOR = TMAKEMONITOR(HEADLESS) creates a training progress monitor, if HEADLESS=true it will create a mock monitor.

    if ~headless
        monitor = trainingProgressMonitor( ...
            Metrics = ["TrainingLoss", "ValidationLoss"]    , ...
            Info    = ["Epoch", "LearningRate", "Iteration"], ...
            XLabel  = "Iteration"                             ...
        );
        groupSubPlot(monitor, "Loss", ["TrainingLoss", "ValidationLoss"]);
    else
        monitor          = struct();
        monitor.Stop     = false;
        monitor.Progress = 0;
    end
end