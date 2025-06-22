function [loss, gradients, state] = PINN_GM_I(net, TRJ, ACC, ~, args)
    % PINN_GM_I OUTDATED

    arguments
        net
        TRJ
        ACC
        ~
        args.trainingMode = true;
    end

    [pPOT, state] = forward(net, TRJ);
    pACC          = -dlgradient(sum(pPOT), TRJ);

    loss = mse(pACC, ACC);

    if args.trainingMode
        gradients = dlgradient(loss, net.Learnables);
    else
        gradients = [];
    end
end