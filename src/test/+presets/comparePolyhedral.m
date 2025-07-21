function [loss, Radius] = comparePolyhedral(TRJ, ACC, ~, pTRJ, pACC, ~)
    % Mean Percentage Error (MPE) loss function for the Polyhedral model.

    arguments
        TRJ
        ACC
        ~
        pTRJ
        pACC
        ~
    end
    
    % Radius
    assert(isequal(TRJ, pTRJ), "Comparing different Trajectory vectors");
    [x, y, z] = deal(TRJ(1, :), TRJ(2, :), TRJ(3, :));
    Radius    = sqrt(x .^ 2 + y .^ 2 + z .^ 2);

    % Loss
    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(~isfinite(PercentLoss)) = 0;
    
    loss = PercentLoss * 100;
end