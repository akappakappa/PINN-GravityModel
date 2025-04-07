function loss = planes(net, Trj, Acc, ~)
    % Forward
    PotPred = forward(net, Trj);

    % Metric
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    PRC     = vecnorm(Acc - AccPred) ./ vecnorm(Acc);
    loss    = mean(PRC, 2) * 100;
end