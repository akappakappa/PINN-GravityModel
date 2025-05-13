function [loss, Radius] = meLoss(net, Trj, Acc, ~)
    % Forward
    [PotPred, Radius] = forward(net, Trj);
    AccPred           = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    loss = vecnorm(Acc - AccPred);
end