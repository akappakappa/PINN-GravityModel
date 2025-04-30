function [loss, gradients, state] = PINN_GM_III_M(net, Trj, Acc, ~, trainingMode)
    % Forward
    [PotPred, Radius, state] = forward(net, Trj);

    % Loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    diff    = AccPred - Acc;
    RMS     = vecnorm(diff);
    MPE     = vecnorm(diff) ./ vecnorm(Acc);
    loss    = mean(RMS + MPE, 2);

    % Gradients
    if trainingMode
        PointLoss = RMS + MPE;
        sig = 5;
        weights = exp(- ((Radius - 1) .^ 2) / (2 * sig ^ 2));
        WeightedLoss = PointLoss .* weights;
        loss = mean(WeightedLoss);
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end