function [loss, gradients, state] = PINN_GM_I(net, TRJ, ACC, ~, args)
    arguments
        net
        TRJ
        ACC
        ~
        args.trainingMode = true;
    end
    % PINN_GM_I  Outdated loss function.

    % Forward
    [pPOT, state] = forward(net, TRJ);
    pACC          = -dlgradient(sum(pPOT, 'all'), TRJ, EnableHigherDerivatives = true);

    % Loss
    loss = mse(pACC, ACC);

    % Gradients
    if args.trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end