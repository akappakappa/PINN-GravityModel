function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, Pot)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % State data generated in the cart2sphLayer
    RotationMatrix = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "RotationMatrix");
    RotationMatrix = RotationMatrix{1};
    ScaleFactor    = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "ScaleFactor");
    ScaleFactor    = ScaleFactor{1};

    % Preprocess Acceleration (rotate)
    Acc = extractdata(Acc);
    Acc = permute(Acc, [1, 3, 2]);
    Acc = pagemtimes(RotationMatrix, Acc);
    Acc = permute(Acc, [1, 3, 2]);

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