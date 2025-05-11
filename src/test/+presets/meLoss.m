function loss = meLoss(net, Trj, Acc, ~)
    % Forward
    PotPred = forward(net, Trj);

    % Metric
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    PRC     = vecnorm(Acc - AccPred);
    loss    = mean(PRC, 2);
end