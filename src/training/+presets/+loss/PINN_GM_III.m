function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, ~, trainingMode)
    % PINN_GM_III  Loss function for the PINN model.
    %   [LOSS, GRADIENTS, STATE] = PINN_GM_III(NET, TRJ, ACC, ~, TRAININGMODE) computes the loss and gradients for the PINN model as the mean of the sum of Mean Percentage Error (MPE) and Mean Error (ME) between the predicted (with automatic differentiation) and the actual acceleration.

    % Forward
    [PotPred, ~, state] = forward(net, Trj);
    AccPred             = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    num  = vecnorm(AccPred - Acc);
    den  = vecnorm(Acc);
    loss = num ./ den;       % MPE
    loss(Inf == loss) = 0;
    loss = loss + num;       % ME
    loss = mean(loss, 2);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end