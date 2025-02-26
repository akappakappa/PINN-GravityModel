assert(1 == exist('dataset', 'var'), 'dataset variable not found');

% Network
layers = [
    featureInputLayer(3, "Name", "featureinput")
    fullyConnectedLayer(32, "Name", "fc1")
    geluLayer("Name", "act1")
    fullyConnectedLayer(32, "Name", "fc2")
    geluLayer("Name", "act2")
    fullyConnectedLayer(32, "Name", "fc3")
    geluLayer("Name", "act3")
    fullyConnectedLayer(32, "Name", "fc4")
    geluLayer("Name", "act4")
    fullyConnectedLayer(32, "Name", "fc5")
    geluLayer("Name", "act5")
    fullyConnectedLayer(32, "Name", "fc6")
    geluLayer("Name", "act6")
    fullyConnectedLayer(1)
];
net = dlnetwork(layers);
net = initialize(net);

% Loss
function [loss, gradients, state] = modelLoss(net, Trj, Acc, Pot)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % Loss, TODO: more big chonky loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    loss = mse(AccPred, Acc);

    % Gradients
    gradients = dlgradient(loss, net.Learnables);
end

function validationLoss = modelLossNoGrad(net, Trj, Acc, Pot)
    % Forward
    PotPred = forward(net, Trj);

    % Loss, TODO: more big chonky loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    validationLoss = mse(AccPred, Acc);
end

% Optimizer
function parameters = sgdStep(parameters, gradients, learnRate)
    parameters = parameters - learnRate .* gradients;
end

% Options
numEpochs             = 2^13;
miniBatchSize         = 2^11;
numObservationsTrain  = datastore.split(1);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;
learnRate             = 2^-8;
learnRateSchedule     = "piecewise";
learnRateDropPeriod   = 2^13;
learnRateDropFactor   = 0.5;
verbose               = true;
verboseFrequency      = 2^6;
validationFrequency   = 2^6;
validationPatience    = 2^4;

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
    MiniBatchSize = miniBatchSize ...
);

mbqVal = minibatchqueue(datastore.validation, ...
    MiniBatchFcn  = @preprocessMiniBatch, ...
    MiniBatchSize = miniBatchSize ...
);

% Monitor
monitor = trainingProgressMonitor( ...
    Metrics = ["TrainingLoss", "ValidationLoss"], ...
    Info    = ["Epoch", "LearningRate", "Iteration"], ...
    XLabel  = "Iteration" ...
);

groupSubPlot(monitor, "Loss", ["TrainingLoss", "ValidationLoss"]);

% Train
epoch = 0;
iteration = 0;
earlyStop = false;
fprintf("|========================================================================================|\n")
fprintf("|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Base Learning  |\n")
fprintf("|         |             |   (hh:mm:ss)   |     Loss     |     Loss     |      Rate       |\n")
fprintf("|========================================================================================|\n")
start = tic;
if isfinite(validationPatience), validationLosses = inf(1, validationPatience); end
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    shuffle(mbq);
    shuffle(mbqVal);

    % Loop over mini-batches
    while hasdata(mbq) && ~monitor.Stop && ~earlyStop
        iteration = iteration + 1;

        % Read mini-batch.
        [Trj, Acc, Pot] = next(mbq);
        if ("auto" == GPU && canUseGPU) || "gpu" == GPU, [Trj, Acc, Pot] = deal(gpuArray(Trj), gpuArray(Acc), gpuArray(Pot)); end

        % Eval and update state
        [loss, gradients, state] = dlfeval(@modelLoss, net, Trj, Acc, Pot);
        net.State = state;

        % Update net
        updateFcn = @(parameters, gradients) sgdStep(parameters, gradients, learnRate);
        net = dlupdate(updateFcn, net, gradients);

        % Update monitor
        recordMetrics(monitor, iteration, TrainingLoss = loss);
        updateInfo(monitor, ...
            Epoch = epoch, ...
            Iteration = iteration, ...
            LearningRate = learnRate ...
        );
        monitor.Progress = 100 * iteration / numIterations;

        % Update Validation Loss
        if iteration == 1 || 0 == mod(iteration, validationFrequency)
            [TrjV, AccV, PotV] = next(mbqVal);
            if ("auto" == GPU && canUseGPU) || "gpu" == GPU, [TrjV, AccV, PotV] = deal(gpuArray(TrjV), gpuArray(AccV), gpuArray(PotV)); end
                validationLoss = dlfeval(@modelLossNoGrad, net, TrjV, AccV, PotV);
            recordMetrics(monitor, iteration, ValidationLoss = validationLoss);

            % Early stopping
            if isfinite(validationPatience)
                validationLosses = [validationLosses validationLoss];
                if min(validationLosses) == validationLosses(1)
                    earlyStop = true;
                else
                    validationLosses(1) = [];
                end
            end
        end

        % Verbose
        if verbose && (1 == iteration || 0 == mod(iteration, verboseFrequency))
            D = duration(0, 0, toc(start), Format = "hh:mm:ss");
            fprintf("| %7d | %11d | %14s | %12.4f | %12.4f | %15.4f |\n", ...
                epoch, iteration, D, loss, validationLoss, learnRate ...
            );
        end
    end
        
    % Update learning rate
    if "piecewise" == learnRateSchedule && 0 == mod(epoch, learnRateDropPeriod)
        learnRate = learnRate * learnRateDropFactor;
    end
end
fprintf("|======================================================================================================================|\n")

% Save
save("src/training/net.mat", "net");