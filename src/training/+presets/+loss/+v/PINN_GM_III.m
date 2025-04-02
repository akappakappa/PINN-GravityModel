function loss = PINN_GM_III(net, Trj, Acc, ~)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % State data generated in the cart2sphLayer
    Radius = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "Radius");
    Radius = Radius{1};

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
    RMS     = vecnorm(AccPred - Acc);
    MPE     = vecnorm(AccPred - Acc) ./ vecnorm(Acc);
    loss    = sum(sum(RMS + MPE, 2) / size(AccPred, 2)) / 3;
end