assert(1 == exist('DEBUG', 'var'), 'you must run this script from src/main.m');
assert(1 == exist('dataset', 'var'), 'dataset variable not found');

% NETWORK
layers = [
    featureInputLayer(3, "Name", "featureinput")
    fullyConnectedLayer(20, "Name", "fc1")
    geluLayer("Name", "act1")
    fullyConnectedLayer(20, "Name", "fc2")
    geluLayer("Name", "act2")
    fullyConnectedLayer(20, "Name", "fc3")
    geluLayer("Name", "act3")
    fullyConnectedLayer(20, "Name", "fc4")
    geluLayer("Name", "act4")
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);
net = initialize(net);
%if true == DEBUG, plot(net), end
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
function [loss, gradients, state] = modelLoss(net, X, T, U)
    % Forward
    [Y, state] = forward(net, X);
    % Loss (TODO: PINN)
    loss = mse(Y, U);
    % Gradients
    gradients = dlgradient(loss, net.Learnables);
end

% Optimizer
function parameters = sgdStep(parameters, gradients, learnRate)
    parameters = parameters - learnRate .* gradients;
end

% Options
numEpochs = 400;
miniBatchSize = 50;
learnRate = 0.0001;

function [X, T, U] = preprocessMiniBatch(dataX, dataT, dataU)
    X = cat(1, dataX{:});
    X = dlarray(X, 'BC');
    T = cat(1, dataT{:});
    T = dlarray(T, 'BC');
    U = cat(1, dataU{:});
    U = dlarray(U, 'BC');
end

mbq = minibatchqueue(datastore.train, ...
    MiniBatchFcn = @preprocessMiniBatch, ...
    MiniBatchSize = miniBatchSize ...
);

% Iterations
numObservationsTrain = datastore.split(1);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

% Monitor
monitor = trainingProgressMonitor( ...
    Metrics = "Loss", ...
    Info = "Epoch", ...
    XLabel = "Iteration" ...
);

% TRAIN
epoch = 0;
iteration = 0;
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    % Shuffle
    shuffle(mbq);
    % Loop over mini-batches
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        % Read mini-batch.
        [X, T, U] = next(mbq);
        % Eval and update state
        [loss, gradients, state] = dlfeval(@modelLoss, net, X, T, U);
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