function [loss, gradients, state] = RadiusWeight(net, TRJ, ACC, ~, args)
    arguments
        net
        TRJ
        ACC
        ~
        args.trainingMode = true;
    end
    % RadiusWeight  Similar to PINN_GM_III, but with adds radius-based weighting to the loss function.
    %   [LOSS, GRADIENTS, STATE] = RADIUSWEIGHT(NET, TRJ, ACC, ~, TRAININGMODE) computes the loss and gradients for the PINN model as the mean of the sum of Mean Percentage Error (MPE) and Mean Error (ME) between the predicted (with automatic differentiation) and the actual acceleration, then weights it by the radius of the trajectory.

    % Forward
    [pPOT, Radius, state] = forward(net, TRJ);
    pACC                  = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(Inf == PercentLoss) = 0;   % Fix division by zero

    loss = AbsoluteLoss + PercentLoss;

    % Radius Weight
    s      = 1;
    weight = (2 ./ (Radius + 2)) .^ s;
    loss   = mean(loss .* weight, 2);

    % Gradients
    if args.trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end