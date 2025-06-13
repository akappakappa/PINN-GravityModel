% This script contains the training loop for the Physics-Informed NN model.
%
% File: runTraining.m
%     entrypoint for Training

executionEnvironment    = "auto";   % Leave as "auto" for GPU training if found, or set to "gpu" or "cpu"
recoverFromCheckpoint   = false;    % Only set to true if you want to recover from a checkpoint on stopped training
headless                = batchStartupOptionUsed;

% Preparations - Data
data      = tLoadData("src/preprocessing/trainingData.mat");
net       = presets.network.PINN_GM_III(data.params);
net       = dlupdate(@double, initialize(net));
modelLoss = dlaccelerate(@presets.loss.PINN_GM_III);
options   = presets.options.PINN_GM_III(data.params.split(1));
clipGrad  = isfield(options, "gradientThresholdMethod") && ~isempty(options.gradientThresholdMethod);
if ("auto" == executionEnvironment || "gpu" == executionEnvironment) && canUseGPU
    net   = dlupdate(@gpuArray, net);
end

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

    % Shuffle training data
    rIdxTrain     = randperm(data.params.split(1));
    data.trainTRJ = data.trainTRJ(:, rIdxTrain);
    data.trainACC = data.trainACC(:, rIdxTrain);
    data.trainPOT = data.trainPOT(:, rIdxTrain);

    numBatches = floor(data.params.split(1) / options.miniBatchSize);
    for batchIdx = 1:numBatches
        iteration = iteration + 1;

        idxTrain        = (1:options.miniBatchSize) + (batchIdx - 1) * options.miniBatchSize;
        [TRJ, ACC, POT] = deal(data.trainTRJ(:, idxTrain), data.trainACC(:, idxTrain), data.trainPOT(:, idxTrain));

        [loss, gradients, net.State]      = dlfeval(modelLoss, net, TRJ, ACC, POT, true);
        if clipGrad
            gradients = dlupdate(@(g) options.gradientThresholdMethod(g, options.gradientThreshold), gradients);
        end
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, options.learnRate);
        
        if ~headless
            recordMetrics(monitor, iteration, TrainingLoss = loss);
        end
        
        % Validation
        if (1 == iteration || 0 == mod(iteration, options.numIterationsPerEpoch)) && ~monitor.Stop
            rIdxValidation  = randperm(data.params.split(2), floor(data.params.split(2) / options.numIterationsPerEpoch) * options.numIterationsPerEpoch);
            [TRJ, ACC, POT] = deal(data.validationTRJ(:, rIdxValidation), data.validationACC(:, rIdxValidation), data.validationPOT(:, rIdxValidation));

            [validationLoss, ~, ~] = dlfeval(modelLoss, net, TRJ, ACC, POT, false);
            
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
save("net", "net");

clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



function data = tLoadData(path)
    % TLOADDATASTORE  Load the data from the specified path.
    %   DATA = TLOADDATASTORE(PATH) loads the data from the specified path, shuffling the training and validation sets.

    data = load(path);

    data.trainTRJ = dlarray(data.trainTRJ, "BC");
    data.trainACC = dlarray(data.trainACC, "BC");
    data.trainPOT = dlarray(data.trainPOT, "BC");

    data.validationTRJ = dlarray(data.validationTRJ, "BC");
    data.validationACC = dlarray(data.validationACC, "BC");
    data.validationPOT = dlarray(data.validationPOT, "BC");
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