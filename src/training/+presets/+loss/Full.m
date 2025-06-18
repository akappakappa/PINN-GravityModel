function [loss, gradients, state] = Full(net, TRJ, ACC, POT, args)
    arguments
        net
        TRJ
        ACC
        POT
        args.trainingMode = true;
    end
    % PINN_GM_III  Loss function for the PINN model.
    %   [LOSS, GRADIENTS, STATE] = PINN_GM_III(NET, TRJ, ACC, ~, TRAININGMODE) computes the loss and gradients for the PINN model as the mean of the sum of Mean Percentage Error (MPE) and Mean Error (ME) between the predicted (with automatic differentiation) and the actual acceleration.

    % Forward
    [pPOT, ~, state] = forward(net, TRJ);
    pACC             = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);
    pLAP             = -dlgradient(sum(pACC, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    PotLoss = vecnorm(pPOT - POT);
    PotLoss = PotLoss + PotLoss ./ vecnorm(POT);
    PotLoss(Inf == PotLoss) = 0;
    AccLoss = vecnorm(pACC - ACC);
    AccLoss = AccLoss + AccLoss ./ vecnorm(ACC);
    AccLoss(Inf == AccLoss) = 0;
    LapLoss = vecnorm(pLAP);
    
    loss = PotLoss + AccLoss + LapLoss;
    loss = mean(loss, 2);

    % Gradients
    if args.trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end