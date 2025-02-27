assert(1 == exist('dataset', 'var'), 'dataset variable not found');
assert(1 == exist('datastore', 'var'), 'datastore variable not found');
executionEnvironment = "auto";

% Presets
net             = initialize(dlnetwork(presets.network.PINN_GM_III));
modelLoss       = @presets.loss.t.PINN_GM_III;
modelLossNoGrad = @presets.loss.v.PINN_GM_III;
opt             = presets.options.PINN_GM_III(datastore.split(1));

% Optimizer
function parameters = sgdStep(parameters, gradients, learnRate)
    parameters = parameters - learnRate .* gradients;
end

% Mini-batch
function [Trj, Acc, Pot] = preprocessMiniBatch(dataTrj, dataAcc, dataPot)
    Trj = cat(1, dataTrj{:});
    Trj = dlarray(Trj, 'BC');
    Acc = cat(1, dataAcc{:});
    Acc = dlarray(Acc, 'BC');
    Pot = cat(1, dataPot{:});
    Pot = dlarray(Pot, 'BC');
end

mbq = minibatchqueue(datastore.train, ...
    MiniBatchFcn  = @preprocessMiniBatch, ...
    MiniBatchSize = opt.miniBatchSize ...
);

mbqVal = minibatchqueue(datastore.validation, ...
    MiniBatchFcn  = @preprocessMiniBatch, ...
    MiniBatchSize = floor(datastore.split(2) / opt.numIterationsPerEpoch) * opt.validationFrequency ...
);

% Monitor
monitor = trainingProgressMonitor( ...
    Metrics = ["TrainingLoss", "ValidationLoss"], ...
    Info    = ["Epoch", "LearningRate", "Iteration"], ...
    XLabel  = "Iteration" ...
);
groupSubPlot(monitor, "Loss", ["TrainingLoss", "ValidationLoss"]);

% Initialize loop
epoch = 0;
iteration = 0;
earlyStop = false;
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
while epoch < opt.numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    shuffle(mbq);
    shuffle(mbqVal);

    % Loop over mini-batches
    while hasdata(mbq) && ~monitor.Stop && ~earlyStop
        iteration = iteration + 1;

        % Read mini-batch.
        [Trj, Acc, Pot] = next(mbq);
        if ("auto" == executionEnvironment && canUseGPU) || "gpu" == executionEnvironment
            [Trj, Acc, Pot] = deal(gpuArray(Trj), gpuArray(Acc), gpuArray(Pot));
        end

        % Eval and update state
        [loss, gradients, state] = dlfeval(modelLoss, net, Trj, Acc, Pot);
        net.State = state;

        % Update net
        updateFcn = @(parameters, gradients) sgdStep(parameters, gradients, opt.learnRate);
        net = dlupdate(updateFcn, net, gradients);

        % Training monitor
        recordMetrics(monitor, iteration, TrainingLoss = loss);
        updateInfo(monitor, Epoch = string(epoch) + " / " + string(opt.numEpochs), Iteration = iteration, LearningRate = opt.learnRate);

        % Validation monitor
        if iteration == 1 || 0 == mod(iteration, opt.validationFrequency)
            [TrjV, AccV, PotV] = next(mbqVal);
            if ("auto" == executionEnvironment && canUseGPU) || "gpu" == executionEnvironment
                [TrjV, AccV, PotV] = deal(gpuArray(TrjV), gpuArray(AccV), gpuArray(PotV));
            end
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
save("src/training/net.mat", "net");