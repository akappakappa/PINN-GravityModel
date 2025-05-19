% This script contains the training loop for the Physics-Informed NN model.
%
% File: runTraining.m
%     entrypoint for Training

executionEnvironment    = "auto";   % Leave as "auto" for GPU training if found, or set to "gpu" or "cpu"
headless                = batchStartupOptionUsed;
recoverFromCheckpoint   = false;    % Only set to true if you want to recover from a checkpoint on stopped training
useGPU                  = ("auto" == executionEnvironment || "gpu" == executionEnvironment) && canUseGPU;

% Preparations - Data
data         = tLoadData("src/preprocessing/trainingData.mat");
net          = dlupdate(@double, initialize(presets.network.PINN_GM_III(data.params.mu, data.params.e)));
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
        [TRJ, ACC, POT]                   = next(tMBQ);
        [loss, gradients, net.State]      = dlfeval(modelLoss, net, TRJ, ACC, POT, true);
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, options.learnRate);
        
        if ~headless
            recordMetrics(monitor, iteration, TrainingLoss = loss);
        end
        
        % Validation
        if (1 == iteration || 0 == mod(iteration, options.numIterationsPerEpoch)) && hasdata(vMBQ) && ~monitor.Stop
            [TRJ, ACC, POT]        = next(vMBQ);
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
    data.train      = shuffle(combine(      ...
        arrayDatastore(data.trainTRJ)     , ...
        arrayDatastore(data.trainACC)     , ...
        arrayDatastore(data.trainPOT)       ...
    ));
    data.validation = shuffle(combine(      ...
        arrayDatastore(data.validationTRJ), ...
        arrayDatastore(data.validationACC), ...
        arrayDatastore(data.validationPOT)  ...
    ));
end

function [tMBQ, vMBQ] = tSetupMinibatchQueues(data, options, executionEnvironment)
    % TSETUPMINIBATCHQUEUES  Setup the mini-batch queues for training and validation.
    %   [TMBQ, VMBQ] = TSETUPMINIBATCHQUEUES(DATA, OPTIONS, EXECUTIONENVIRONMENT) sets up the mini-batch queues for training and validation, given OPTIONS and EXECUTIONENVIRONMENT.
    
    tMBQ = minibatchqueue(data.train                                           , ...
        MiniBatchFcn      = @(TRJ, ACC, POT) preprocessMiniBatch(TRJ, ACC, POT), ...
        MiniBatchSize     = options.miniBatchSize                              , ...
        OutputEnvironment = executionEnvironment                               , ...
        MiniBatchFormat   = 'BC'                                               , ...
        OutputCast        = "double"                                             ...
    );
    
    vMBQ = minibatchqueue(data.validation                                                                              , ...
        MiniBatchFcn      = @(TRJ, ACC, POT) preprocessMiniBatch(TRJ, ACC, POT)                                        , ...
        MiniBatchSize     = floor(data.params.split(2) / options.numIterationsPerEpoch) * options.numIterationsPerEpoch, ...
        OutputEnvironment = executionEnvironment                                                                       , ...
        MiniBatchFormat   = 'BC'                                                                                       , ...
        OutputCast        = "double"                                                                                     ...
    );



    function [TRJ, ACC, POT] = preprocessMiniBatch(TRJ, ACC, POT)
        % PREPROCESSMINIBATCH  Preprocess the mini-batch data.
        %   [TRJ, ACC, POT] = PREPROCESSMINIBATCH(TRJ, ACC, POT) preprocesses the mini-batch data.

        TRJ = cat(1, TRJ{:});
        ACC = cat(1, ACC{:});
        POT = cat(1, POT{:});
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