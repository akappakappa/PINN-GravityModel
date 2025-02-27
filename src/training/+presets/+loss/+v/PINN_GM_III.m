function loss = PINN_GM_III(net, Trj, Acc, Pot)
    % Forward
    PotPred = forward(net, Trj);

    % Loss, TODO: more big chonky loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    loss = mse(AccPred, Acc);
end