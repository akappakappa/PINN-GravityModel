function loss = mpeLoss(net, Trj, Acc, ~)
    % Forward
    PotPred = forward(net, Trj);

    % Metric
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    PRC     = vecnorm(Acc - AccPred) ./ vecnorm(Acc) * 100;
    loss    = mean(PRC, 2);
end