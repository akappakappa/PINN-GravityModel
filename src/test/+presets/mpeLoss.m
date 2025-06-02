function [loss, Radius] = mpeLoss(net, TRJ, ACC, ~)
    % mpeLoss  Mean Percentage Error (MPE) loss function for the PINN model.
    %   [LOSS, RADIUS] = MPELOSS(NET, TRJ, ACC, ~) computes the loss for the PINN model as the mean of the sum of Mean Percentage Error (MPE) between the predicted (with automatic differentiation) and the actual acceleration.

    % Forward
    [pPOT, Radius] = forward(net, TRJ);
    pACC           = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(Inf == PercentLoss) = 0;   % Fix division by zero
    
    loss = PercentLoss * 100;
end