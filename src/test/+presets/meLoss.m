function [loss, Radius] = meLoss(net, TRJ, ACC, ~)
    % Forward
    [pPOT, Radius] = forward(net, TRJ);
    pACC           = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    AbsoluteLoss = vecnorm(ACC - pACC);
    
    loss = AbsoluteLoss;
end