%assert(exist('DEBUG', 'var') == 1, 'you must run this script from src/main.m');
%assert(exist('dataset', 'var') == 1, 'dataset variable not found');
%disp('Asserts passed');

% NETWORK
%layers = [
%    featureInputLayer(3, "Name", "in")
%    fullyConnectedLayer(20, "Name", "fc1")
%    fullyConnectedLayer(20, "Name", "fc2")
%    fullyConnectedLayer(20, "Name", "fc3")
%    fullyConnectedLayer(1, "Name", "fc4")];
layers = [
    featureInputLayer(3,"Name","featureinput")
    geluLayer("Name","gelu")
    fullyConnectedLayer(20,"Name","fc")
    geluLayer("Name","tanh")
    fullyConnectedLayer(20,"Name","fc_1")
    geluLayer("Name","tanh_1")
    fullyConnectedLayer(20,"Name","fc_2")
    geluLayer("Name","tanh_2")
    fullyConnectedLayer(20,"Name","fc_3")
    geluLayer("Name","tanh_3")
    fullyConnectedLayer(1,"Name","fc_4")];

net = dlnetwork(layers);
net = initialize(net);
%if true == DEBUG, plot(net), end

% OPTIONS
%options = trainingOptions("adam", ...
%    "MaxEpochs", 400, ...
%    "MiniBatchSize", 4, ...
%    "InitialLearnRate", 0.0001, ...
%    "Plots", "training-progress");

% TRAIN
%netTrained = trainnet(xTrain, yTrain, net, "mse", options);

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
    X = dlarray(X,'BC');
    T = cat(1, dataT{:});
    T = dlarray(T,'BC');
    U = cat(1, dataU{:});
    U = dlarray(U,'BC');
end

mbq = minibatchqueue(trainingSet, ...
    MiniBatchFcn= @preprocessMiniBatch, ...
    MiniBatchSize= miniBatchSize);

% Iterations
numObservationsTrain = 96900; % size(xTrain, 1);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

% Monitor
monitor = trainingProgressMonitor( ...
    Metrics = "Loss", ...
    Info = "Epoch", ...
    XLabel = "Iteration");

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
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch);
        monitor.Progress = 100 * iteration/numIterations;
    end
end