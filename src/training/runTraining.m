executionEnvironment = "auto";
useGPU               = ("auto" == executionEnvironment && canUseGPU) || "gpu" == executionEnvironment;

% Load datastore
ds.params     = readstruct("src/preprocessing/datastore/params.json");
ds.train      = shuffle(combine( ...
    arrayDatastore(readmatrix("src/preprocessing/datastore/train/Trj.csv")), ...
    arrayDatastore(readmatrix("src/preprocessing/datastore/train/Acc.csv")), ...
    arrayDatastore(readmatrix("src/preprocessing/datastore/train/Pot.csv")) ...
));
ds.validation = shuffle(combine( ...
    arrayDatastore(readmatrix("src/preprocessing/datastore/validation/Trj.csv")), ...
    arrayDatastore(readmatrix("src/preprocessing/datastore/validation/Acc.csv")), ...
    arrayDatastore(readmatrix("src/preprocessing/datastore/validation/Pot.csv")) ...
));

% Presets
net             = initialize(presets.network.PINN_GM_III(ds.params.mu, ds.params.e));
modelLoss       = dlaccelerate(@presets.loss.t.PINN_GM_III);
modelLossNoGrad = dlaccelerate(@presets.loss.v.PINN_GM_III);
opt             = presets.options.PINN_GM_III(ds.params.split(1));
if useGPU
    net = dlupdate(@gpuArray, net);
end

% Mini-batch
function [Trj, Acc, Pot] = preprocessMiniBatch(Trj, Acc, Pot, useGPU)
    [Trj, Acc, Pot] = deal(cat(1, Trj{:}), cat(1, Acc{:}), cat(1, Pot{:}));
    if useGPU
        [Trj, Acc, Pot] = deal(gpuArray(Trj), gpuArray(Acc), gpuArray(Pot));
    end
    [Trj, Acc, Pot] = deal(dlarray(Trj, 'BC'), dlarray(Acc, 'BC'), dlarray(Pot, 'BC'));
end

mbq = minibatchqueue(ds.train, ...
    MiniBatchFcn  = @(Trj, Acc, Pot) preprocessMiniBatch(Trj, Acc, Pot, useGPU), ...
    MiniBatchSize = opt.miniBatchSize ...
);

mbqVal = minibatchqueue(ds.validation, ...
    MiniBatchFcn  = @(Trj, Acc, Pot) preprocessMiniBatch(Trj, Acc, Pot, useGPU), ...
    MiniBatchSize = floor(ds.params.split(2) / opt.numIterationsPerEpoch) * floor(opt.numIterationsPerEpoch / opt.validationFrequency) ...
);

% Monitor
monitor = trainingProgressMonitor( ...
    Metrics = ["TrainingLoss", "ValidationLoss"], ...
    Info    = ["Epoch", "LearningRate", "Iteration"], ...
    XLabel  = "Iteration" ...
);
groupSubPlot(monitor, "Loss", ["TrainingLoss", "ValidationLoss"]);

% Initialize loop
epoch         = 0;
iteration     = 0;
averageGrad   = [];
averageSqGrad = [];
earlyStop     = false;
if opt.verbose
    fprintf("|========================================================================================|\n");
    fprintf("|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Base Learning  |\n");
    fprintf("|         |             |   (hh:mm:ss)   |     Loss     |     Loss     |      Rate       |\n");
    fprintf("|========================================================================================|\n");
end
start = tic;
if isfinite(opt.validationPatience)
    validationLosses = inf(1, opt.validationPatience);
end

% Loop
while epoch < opt.numEpochs && ~monitor.Stop && ~earlyStop
    epoch = epoch + 1;
    shuffle(mbq);
    shuffle(mbqVal);

    % Loop over mini-batches
    while hasdata(mbq) && ~monitor.Stop && ~earlyStop && iteration < opt.numIterations
        iteration = iteration + 1;

        % Read mini-batch.
        [Trj, Acc, Pot] = next(mbq);

        % Eval and update state
        [loss, gradients, state] = dlfeval(modelLoss, net, Trj, Acc, Pot);
        net.State = state;

        % Update net with Adam optimizer
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, opt.learnRate);

        % Training monitor
        recordMetrics(monitor, iteration, TrainingLoss = loss);
        updateInfo(monitor, Epoch = string(epoch) + " / " + string(opt.numEpochs), Iteration = iteration, LearningRate = opt.learnRate);

        % Validation monitor
        if iteration == 1 || 0 == mod(iteration, floor(opt.numIterationsPerEpoch / opt.validationFrequency))
            [TrjV, AccV, PotV] = next(mbqVal);
            validationLoss = dlfeval(modelLossNoGrad, net, TrjV, AccV, PotV);
            recordMetrics(monitor, iteration, ValidationLoss = validationLoss);

            % Early stopping
            if isfinite(opt.validationPatience)
                validationLosses = [validationLosses validationLoss];
                if min(validationLosses) == validationLosses(1)
                    earlyStop = true;
                else
                    validationLosses(1) = [];
                end
            end
        end
        monitor.Progress = 100 * iteration / opt.numIterations;

        % Verbose
        if opt.verbose && (1 == iteration || 0 == mod(iteration, opt.verboseFrequency))
            D = duration(0, 0, toc(start), Format = "hh:mm:ss");
            fprintf("| %7d | %11d | %14s | %12.4f | %12.4f | %15.4f |\n", ...
                epoch, iteration, D, loss, validationLoss, opt.learnRate ...
            );
        end
    end
        
    % Update learning rate
    if "piecewise" == opt.learnRateSchedule && 0 == mod(epoch, opt.learnRateDropPeriod)
        opt.learnRate = opt.learnRate * opt.learnRateDropFactor;
    end
end
if opt.verbose, fprintf("|========================================================================================|\n"); end

% Save
save("net", "net");