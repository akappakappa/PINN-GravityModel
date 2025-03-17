function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, Pot)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % Extract data from cart2sphLayer
    dataLayer = net.Layers(2);
    RotationMatrix = dataLayer.RotationMatrix;
    ScaleFactor = dataLayer.ScaleFactor;

    % Preprocess Acceleration (rotate)
    Acc = permute(Acc, [2, 3, 1]);
    Acc = pagemtimes(RotationMatrix, Acc);
    Acc = permute(Acc, [3, 1, 2]);

    % Preprocess Potential (proxy)
    PotPred = PotPred ./ ScaleFactor;

    % Loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    loss = mse(AccPred, Acc);
    %RMS = rmse(AccPred, Acc);
    %MPE = mape(AccPred, Acc) / 100;
    %loss = RMS + MPE;

    % Gradients
    gradients = dlgradient(loss, net.Learnables);
end