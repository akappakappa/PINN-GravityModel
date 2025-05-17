function [loss, gradients, state] = PINN_GM_I(net, Trj, Acc, ~, trainingMode)
    % PINN_GM_I  Outdated loss function.

    % Forward
    [PotPred, state] = forward(net, Trj);
    AccPred          = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    loss = mse(AccPred, Acc);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end