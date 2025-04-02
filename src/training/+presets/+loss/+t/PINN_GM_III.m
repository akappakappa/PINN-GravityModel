function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, ~)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % State data generated in the cart2sphLayer
    RotationMatrix = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "RotationMatrix");
    RotationMatrix = RotationMatrix{1};
    Radius         = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "Radius");
    Radius         = Radius{1};

    % Preprocess Acceleration (rotate)
    %Acc = extractdata(Acc);
    %Acc = permute(Acc, [1, 3, 2]);
    %Acc = pagemtimes(RotationMatrix, Acc);
    %Acc = permute(Acc, [1, 3, 2]);
    %Acc = dlarray(Acc, 'CB');

    % Preprocess Potential (proxy)
    ScaleFactor                   = Radius;
    ScaleFactor(ScaleFactor <= 1) = 1;
    PotPred                       = PotPred ./ ScaleFactor;

    % Preprocess Potential (boundary conditions) TODO: PotBC ??
    [rref, PotBC] = deal(Inf, 0);
    k             = 2;
    h             = (1 + tanh(k * (Radius - rref))) / 2;
    wnn           = 1 - h;
    wbc           = h;
    PotPred       = wnn .* PotPred + wbc .* PotBC;

    % Loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    RMS  = vecnorm(AccPred - Acc);
    MPE  = vecnorm(AccPred - Acc) ./ vecnorm(Acc);
    loss = sum(sum(RMS + MPE, 2) / size(AccPred, 2)) / 3;

    % Gradients
    gradients = dlgradient(loss, net.Learnables);
end