function loss = generalized(net, Trj, Acc, ~)
    % Forward
    PotPred = forward(net, Trj);

    % Metric
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    diff    = AccPred - Acc;
    RMS     = vecnorm(diff);
    MPE     = vecnorm(diff) ./ vecnorm(Acc);
    loss    = mean(RMS + MPE, 2);
end