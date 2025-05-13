function [loss, gradients, state] = WeightedRadius(net, Trj, Acc, ~, trainingMode)
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

    % Weighted loss
    sig        = 5;
    weights    = exp(-((Radius - 1) .^ 2) / (2 * sig ^ 2));
    loss       = mean(loss .* weights, 2);

    % Gradients
    if trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end