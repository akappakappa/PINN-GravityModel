function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, ~, trainingMode)
    % Forward
    [PotPred, ~, state] = forward(net, Trj);
    AccPred             = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    num        = vecnorm(AccPred - Acc);
    den        = vecnorm(Acc);
    loss       = zeros(size(Acc, 2), 1);
    mask       = den ~= 0;
    loss(mask) = num(mask) ./ den(mask);   % MPE
    loss       = loss + num;               % ME
    loss       = mean(loss, 2);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end