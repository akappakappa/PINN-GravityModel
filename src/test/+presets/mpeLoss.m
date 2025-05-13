function [loss, Radius] = mpeLoss(net, Trj, Acc, ~)
    % Forward
    [PotPred, Radius] = forward(net, Trj);
    AccPred           = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);

    % Loss
    num  = vecnorm(AccPred - Acc);
    den  = vecnorm(Acc);
    loss = num ./ den * 100;   % MPE
    loss(Inf == loss) = 0;
end