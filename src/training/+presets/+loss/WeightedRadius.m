function [loss, gradients, state] = WeightedRadius(net, Trj, Acc, ~, trainingMode)
    % WeightedRadius  Similar to PINN_GM_III, but with adds radius-based weighting to the loss function.
    %   [LOSS, GRADIENTS, STATE] = WEIGHTEDRADIUS(NET, TRJ, ACC, ~, TRAININGMODE) computes the loss and gradients for the PINN model as the mean of the sum of Mean Percentage Error (MPE) and Mean Error (ME) between the predicted (with automatic differentiation) and the actual acceleration, then weights it by the radius of the trajectory.

    % Forward
    [PotPred, Radius, state] = forward(net, Trj);
    AccPred                  = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    num  = vecnorm(AccPred - Acc);
    den  = vecnorm(Acc);
    loss = num ./ den;       % MPE
    loss(Inf == loss) = 0;
    loss = loss + num;       % ME

    % Weighted loss
    sig     = 5;
    weights = exp(-((Radius - 1) .^ 2) / (2 * sig ^ 2));
    loss    = mean(loss .* weights, 2);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end