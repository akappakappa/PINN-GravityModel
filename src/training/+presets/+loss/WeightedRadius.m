function [loss, gradients, state] = WeightedRadius(net, TRJ, ACC, ~, args)
    arguments
        net
        TRJ
        ACC
        ~
        args.trainingMode = true;
    end
    % WeightedRadius  Similar to PINN_GM_III, but with adds radius-based weighting to the loss function.
    %   [LOSS, GRADIENTS, STATE] = WEIGHTEDRADIUS(NET, TRJ, ACC, ~, TRAININGMODE) computes the loss and gradients for the PINN model as the mean of the sum of Mean Percentage Error (MPE) and Mean Error (ME) between the predicted (with automatic differentiation) and the actual acceleration, then weights it by the radius of the trajectory.

    % Forward
    [pPOT, Radius, state] = forward(net, TRJ);
    pACC                  = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(Inf == PercentLoss) = 0;   % Fix division by zero

    loss = AbsoluteLoss + PercentLoss;
    sig     = 5;
    weights = exp(-((Radius - 1) .^ 2) / (2 * sig ^ 2));
    loss    = mean(loss .* weights, 2);

    % Gradients
    if args.trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end