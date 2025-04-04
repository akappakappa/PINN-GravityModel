function loss = PINN_GM_III(net, Trj, Acc, ~, mu, e)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % State data generated in the cart2sphLayer
    Radius = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "Radius");
    Radius = Radius{1};

    % Preprocess Potential (proxy)
    ScaleFactor                   = Radius;
    ScaleFactor(ScaleFactor <= 1) = 1;
    PotPred                       = PotPred ./ ScaleFactor;

    % Low-Fidelity Potential
    fx    = 0;                      % Extra (optional) terms from Spherical Harmonics model
    PotLF = -(mu ./ Radius + fx);
    function weight = Transition(radius, reference, smoothing)
        weight = (1 + tanh(smoothing * (radius - reference))) / 2;
    end

    % Fusing with Analytical model
    refFusion         = 1 + e;
    smoothFusion      = 0.5;                                          % Slower transition
    weightLowFidelity = Transition(Radius, refFusion, smoothFusion);   % Low-Fidelity model gradually more important around 1+e
    PotFused          = PotPred + weightLowFidelity .* PotLF;

    % Boundary Conditions
    refBounds     = 10;                                           % 10R = max altitude of the training dataset
    smoothBounds  = 2;                                            % Faster transition
    weightBounds  = Transition(Radius, refBounds, smoothBounds);   % Smooth transition from Network to Boundary Conditions around 10R
    weightNetwork = 1 - weightBounds;
    
    PotPred = weightNetwork .* PotFused + weightBounds .* PotLF;

    % Loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    diff    = AccPred - Acc;
    RMS     = vecnorm(diff);
    MPE     = vecnorm(diff) ./ vecnorm(Acc);
    loss    = sum(mean(RMS, 2) + mean(MPE, 2), 2);
    %loss    = sum(RMS + MPE, 2) / size(AccPred, 2);
end