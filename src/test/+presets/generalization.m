function loss = generalization(net, Trj, Acc, ~)
    % Forward
    PotPred = forward(net, Trj);

    % Metric
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    PRC     = vecnorm(Acc - AccPred) ./ vecnorm(Acc);
    loss    = mean(PRC, 2) * 100;
    %diff    = AccPred - Acc;
    %RMS     = vecnorm(diff);
    %MPE     = vecnorm(diff) ./ vecnorm(Acc);
    %loss    = mean(RMS + MPE, 2);
end