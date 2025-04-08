function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, ~, trainingMode)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % Loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    diff    = AccPred - Acc;
    RMS     = vecnorm(diff);
    MPE     = vecnorm(diff) ./ vecnorm(Acc);
    loss    = mean(RMS + MPE, 2);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end