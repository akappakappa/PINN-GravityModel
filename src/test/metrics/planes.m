function metric = planes(net, Trj, Acc, ~)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % State data generated in the cart2sphLayer
    Radius = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "Radius");
    Radius = Radius{1};

    % Preprocess Potential (proxy)
    ScaleFactor                   = Radius;
    ScaleFactor(ScaleFactor <= 1) = 1;
    PotPred                       = PotPred ./ ScaleFactor;

    % Metric
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    PRC     = vecnorm(Acc - AccPred) ./ vecnorm(Acc);
    metric  = sum(PRC) / size(PRC, 2);
end