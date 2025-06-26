function [loss, gradients, state] = PINN_GM_III(net, TRJ, ACC, ~, args)
    % PINN_GM_III ME + MPE
    % Compute Mean Error (ME) + Mean Percentage Error (MPE) loss between true accelerations and predicted ones.
    % Predicted acceleration is obtained with autodiff as the gradients of the predicted potential (Network's output) wrt input coordinates.

    arguments
        net
        TRJ
        ACC
        ~
        args.trainingMode = true;
    end

    [pPOT, ~, state] = forward(net, TRJ);
    pACC             = -dlgradient(sum(pPOT), TRJ);

    AbsoluteLoss = vecnorm(pACC - ACC);
    PercentLoss  = AbsoluteLoss ./ vecnorm(ACC);
    PercentLoss(~isfinite(PercentLoss)) = 0;

    loss = AbsoluteLoss + PercentLoss;
    loss = mean(loss, 2);

    if args.trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end