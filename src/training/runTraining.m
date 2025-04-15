executionEnvironment  = "auto";
headless              = false;
recoverFromCheckpoint = false;
useGPU                = ("auto" == executionEnvironment && canUseGPU) || "gpu" == executionEnvironment;

% Preparations - Data
data                      = tLoadDatastore("src/preprocessing/datastore");
[net, modelLoss, options] = tLoadPresets(data, useGPU);
[tMBQ, vMBQ]              = tSetupMinibatchQueues(data, options, useGPU);

% Preparations - Loop
epoch         = 0;
iteration     = 0;
averageGrad   = [];
averageSqGrad = [];
bestNet       = net;

% Recovering from checkpoint
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
return

function data = tLoadDatastore(path)
    data            = struct();
    data.params     = readstruct(path + "/params.json");
    data.train      = shuffle(combine( ...
        arrayDatastore(readmatrix(path + "/train/Trj.csv"     )), ...
        arrayDatastore(readmatrix(path + "/train/Acc.csv"     )), ...
        arrayDatastore(readmatrix(path + "/train/Pot.csv"     ))  ...
    ));
    data.validation = shuffle(combine( ...
        arrayDatastore(readmatrix(path + "/validation/Trj.csv")), ...
        arrayDatastore(readmatrix(path + "/validation/Acc.csv")), ...
        arrayDatastore(readmatrix(path + "/validation/Pot.csv"))  ...
    ));
end

function [network, loss, options] = tLoadPresets(data, useGPU)    
    network = initialize(presets.network.PINN_GM_III(data.params.mu, data.params.e));
    loss    = dlaccelerate(@presets.loss.PINN_GM_III);
    options = presets.options.PINN_GM_III(data.params.split(1));
    if useGPU
        network = dlupdate(@gpuArray, network);
    end
end

function [tMBQ, vMBQ] = tSetupMinibatchQueues(data, options, useGPU)
    function [Trj, Acc, Pot] = preprocessMiniBatch(Trj, Acc, Pot, useGPU)
        [Trj, Acc, Pot]     = deal(cat(1, Trj{:})    , cat(1, Acc{:})    , cat(1, Pot{:})    );
        if useGPU
            [Trj, Acc, Pot] = deal(gpuArray(Trj)     , gpuArray(Acc)     , gpuArray(Pot)     );
        end
        [Trj, Acc, Pot]     = deal(dlarray(Trj, 'BC'), dlarray(Acc, 'BC'), dlarray(Pot, 'BC'));
    end
    
    tMBQ = minibatchqueue(data.train                                               , ...
        MiniBatchFcn  = @(Trj, Acc, Pot) preprocessMiniBatch(Trj, Acc, Pot, useGPU), ...
        MiniBatchSize = options.miniBatchSize                                        ...
    );
    
    vMBQ = minibatchqueue(data.validation                                                                          , ...
        MiniBatchFcn  = @(Trj, Acc, Pot) preprocessMiniBatch(Trj, Acc, Pot, useGPU)                                , ...
        MiniBatchSize = floor(data.params.split(2) / options.numIterationsPerEpoch) * options.numIterationsPerEpoch  ...
    );
end

function monitor = tMakeMonitor(headless)
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