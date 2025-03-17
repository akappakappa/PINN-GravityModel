function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, Pot)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % Loss, TODO: more big chonky loss
    ri = Trj(:, 1);
    re = Trj(:, 2);
    ri(ri = 1) = 0;
    re(re = 1) = 0;
    re(re ~= 0) = 1 / re;
    scalingVAL = ri + re;
    scalingVAL(scalingVAL <= const.starTRJ) = 1;
    PotPred = PotPred / scalingVAL;

    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    RMS = rmse(AccPred, Acc);
    MPE = mape(AccPred, Acc) / 100;
    loss = RMS + MPE;

    % Gradients
    gradients = dlgradient(loss, net.Learnables);
end