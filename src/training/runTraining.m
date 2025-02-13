%assert(exist('DEBUG', 'var') == 1, 'you must run this script from src/main.m');
%assert(exist('dataset', 'var') == 1, 'dataset variable not found');
%disp('Asserts passed');

% NETWORK
layers = [
    featureInputLayer(3, "Name", "in")
    fullyConnectedLayer(20, "Name", "fc1")
    fullyConnectedLayer(20, "Name", "fc2")
    fullyConnectedLayer(1, "Name", "fc3")
    sigmoidLayer("Name", "out")];
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
function [loss, gradients, state] = modelLoss(net, X, T)
    % Forward
    [Y, state] = forward(net, X);
    % Loss (TODO: PINN)
    loss = mse(Y, T);
    % Gradients\
    gradients = dlgradient(loss, net.Learnables);
end

% Optimizer
function parameters = sgdStep(parameters, gradients, learnRate)
    parameters = parameters - learnRate .* gradients;
end

% Options
numEpochs = 400;
miniBatchSize = 4;
learnRate = 0.0001;

function [X, T, Y] = preprocessMiniBatch(dataX, dataT, dataY)
    X = cat(1, dataX{:});
    X = dlarray(X,'BC');
    T = cat(1, dataT{:});
    T = dlarray(T,'BC');
    Y = cat(1, dataY{:});
    Y = dlarray(Y,'BC');
end

mbq = minibatchqueue(trainingSet, ...
    MiniBatchFcn= @preprocessMiniBatch, ...
    MiniBatchSize= miniBatchSize);

% Iterations
numObservationsTrain = 39; % size(xTrain, 1);
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
        [X, T, Y] = next(mbq);
        % Eval and update state
        [loss, gradients, state] = dlfeval(@modelLoss, net, X, Y);
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