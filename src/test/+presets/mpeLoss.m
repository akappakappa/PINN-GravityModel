function [loss, Radius] = mpeLoss(net, TRJ, ACC, ~)
    % Mean Percentage Error (MPE) loss function for the PINN model.

    arguments
        net
        TRJ
        ACC
        ~
    end

    % Forward
    [pPOT, Radius] = forward(net, TRJ);
    pACC           = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(~isfinite(PercentLoss)) = 0;
    
    loss = PercentLoss * 100;
end