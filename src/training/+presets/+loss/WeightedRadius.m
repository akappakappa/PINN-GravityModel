function [loss, gradients, state] = WeightedRadius(net, TRJ, ACC, ~, args)
    % WeightedRadius weight * (ME + MPE)
    % Compute weighted Mean Error (ME) + Mean Percentage Error (MPE) loss between true accelerations and predicted ones.
    % Predicted acceleration is obtained with autodiff as the gradients of the predicted potential (Network's output) wrt input coordinates.
    % Weights depend on the radius of each sample in the mini-batch.

    arguments
        net
        TRJ
        ACC
        ~
        args.trainingMode = true;
    end

    [pPOT, Radius, state] = forward(net, TRJ);
    pACC                  = -dlgradient(sum(pPOT), TRJ);

    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(Inf == PercentLoss) = 0;

    loss = AbsoluteLoss + PercentLoss;

    ampl     = 0.5;
    sigL     = 1;
    sigR     = 2;
    peak     = 3;
    
    Common = -(Radius - peak) .^ 2 ./ 2;
    WL     = 1 + ampl .* exp(Common ./ (sigL ^ 2));
    WR     = 1 + ampl .* exp(Common ./ (sigR ^ 2));

    M = Radius < peak;
    W = M .* WL + ~M .* WR;

    loss  = mean(loss .* W, 2);

    if args.trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end