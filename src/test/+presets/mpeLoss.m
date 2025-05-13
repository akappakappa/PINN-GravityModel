function [loss, Radius] = mpeLoss(net, Trj, Acc, ~)
    % Forward
    [PotPred, Radius] = forward(net, Trj);
    AccPred           = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    num        = vecnorm(AccPred - Acc);
    den        = vecnorm(Acc);
    loss       = zeros(size(Acc, 2), 1);
    mask       = den ~= 0;
    loss(mask) = num(mask) ./ den(mask) * 100;
end