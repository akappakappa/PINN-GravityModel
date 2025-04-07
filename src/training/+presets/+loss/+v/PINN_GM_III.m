function loss = PINN_GM_III(net, Trj, Acc, ~)
    % Forward
    PotPred = forward(net, Trj);

    % Loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    diff    = AccPred - Acc;
    RMS     = vecnorm(diff);
    MPE     = vecnorm(diff) ./ vecnorm(Acc);
    loss    = mean(RMS + MPE, 2);
end