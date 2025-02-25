assert(1 == exist('DEBUG', 'var'), 'you must run this script from src/main.m');
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

% Conditions
if true == PLOTNET, plot(net), end
if true == EZMODE
    options = trainingOptions( ...
        "adam", ...
        "MaxEpochs", 400, ...
        "MiniBatchSize", 4, ...
        "InitialLearnRate", 0.0001, ...
        "Plots", "training-progress" ...
    );
    netTrained = trainnet(xTrain, yTrain, net, "mse", options);
    return;
end

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

% Optimizer
function parameters = sgdStep(parameters, gradients, learnRate)
    parameters = parameters - learnRate .* gradients;
end

% Options
numEpochs     = 2^13;
miniBatchSize = 2^11;
%schedule = piecewiseLearnRate( ...
%    "DropFactor", 0.5, ...
%    "Period", 2^13, ...
%    "FrequencyUnit", "epoch", ...
%    "NumSteps", numEpochs ...
%);
%learnRate     = 2^-8;
learnRate     = 0.0001;

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

% Iterations
numObservationsTrain  = datastore.split(1);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;

% Monitor
monitor = trainingProgressMonitor( ...
    Metrics = "Loss", ...
    Info    = "Epoch", ...
    XLabel  = "Iteration" ...
);

% Train
epoch = 0;
iteration = 0;
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    shuffle(mbq);

    % Loop over mini-batches
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        % Read mini-batch.
        [Trj, Acc, Pot] = next(mbq);

        % Eval and update state
        [loss, gradients, state] = dlfeval(@modelLoss, net, Trj, Acc, Pot);
        net.State = state;

        % Update net
        updateFcn = @(parameters, gradients) sgdStep(parameters, gradients, learnRate);
        net = dlupdate(updateFcn, net, gradients);

        % Update monitor
        recordMetrics(monitor, iteration, Loss = loss);
        updateInfo(monitor, Epoch = epoch);
        monitor.Progress = 100 * iteration / numIterations;
    end
end