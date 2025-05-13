function [loss, gradients, state] = PINN_GM_III(net, Trj, Acc, ~, trainingMode)
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