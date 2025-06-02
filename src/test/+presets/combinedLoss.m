function [loss, Radius] = combinedLoss(net, TRJ, ACC, ~)
    % combinedLoss  Mean Error (ME) + Mean Percentage Error (MPE) loss function for the PINN model.
    %   [LOSS, RADIUS] = MPELOSS(NET, TRJ, ACC, ~) computes the loss for the PINN model as the mean of the sum of Mean Error (ME) + Mean Percentage Error (MPE) between the predicted (with automatic differentiation) and the actual acceleration.

    % Forward
    [pPOT, Radius] = forward(net, TRJ);
    pACC           = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(Inf == PercentLoss) = 0;   % Fix division by zero
    
    loss = AbsoluteLoss + PercentLoss;
end