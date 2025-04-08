function [loss, gradients, state] = PINN_GM_I(net, Trj, Acc, ~, trainingMode)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % Loss, TODO: more big chonky loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    loss    = mse(AccPred, Acc);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end