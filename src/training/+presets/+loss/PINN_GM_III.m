function [loss, gradients, state] = PINN_GM_III(net, TRJ, ACC, ~, trainingMode)
    % PINN_GM_III  Loss function for the PINN model.
    %   [LOSS, GRADIENTS, STATE] = PINN_GM_III(NET, TRJ, ACC, ~, TRAININGMODE) computes the loss and gradients for the PINN model as the mean of the sum of Mean Percentage Error (MPE) and Mean Error (ME) between the predicted (with automatic differentiation) and the actual acceleration.

    % Forward
    [pPOT, ~, state] = forward(net, TRJ);
    pACC             = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(Inf == PercentLoss) = 0;   % Fix division by zero

    loss = AbsoluteLoss + PercentLoss;
    loss = mean(loss, 2);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end