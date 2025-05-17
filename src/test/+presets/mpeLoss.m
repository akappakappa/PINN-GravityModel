function [loss, Radius] = mpeLoss(net, Trj, Acc, ~)
    % mpeLoss  Mean Percentage Error (MPE) loss function for the PINN model.
    %   [LOSS, RADIUS] = MPELOSS(NET, TRJ, ACC, ~) computes the loss for the PINN model as the mean of the sum of Mean Percentage Error (MPE) between the predicted (with automatic differentiation) and the actual acceleration.

    % Forward
    [PotPred, Radius] = forward(net, Trj);
    AccPred           = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    num  = vecnorm(AccPred - Acc);
    den  = vecnorm(Acc);
    loss = num ./ den * 100;   % MPE
    loss(Inf == loss) = 0;
end