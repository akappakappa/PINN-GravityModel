function loss = PINN_GM_III(net, Trj, Acc, ~)
    % Forward
    [PotPred, state] = forward(net, Trj);

    % State data generated in the cart2sphLayer
    Radius = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "Radius");
    Radius = Radius{1};

    % Preprocess Potential (proxy)
    ScaleFactor                   = Radius;
    ScaleFactor(ScaleFactor <= 1) = 1;
    PotPred                       = PotPred ./ ScaleFactor;

    % Preprocess Potential (boundary conditions)
    g       = 6.67430e-11;
    vol     = 2525994603183.156;   % from 8k file in dataset https://github.com/MartinAstro/GravNN
    density = 2670;                % https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=eros
    mu      = g * vol * density;
    fx      = 0;                   % Extra (optional) terms from Spherical Harmonics model
    
    PotBC = mu ./ Radius + fx;
    rref  = 10;                    % 10R = max altitude of the training dataset

    k   = 2;
    h   = (1 + tanh(k * (Radius - rref))) / 2;
    wnn = 1 - h;
    wbc = h;

    PotPred = wnn .* PotPred + wbc .* PotBC;

    % Loss
    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    diff    = AccPred - Acc;
    RMS     = vecnorm(diff);
    MPE     = vecnorm(diff) ./ vecnorm(Acc);
    loss    = sum(mean(RMS, 2) + mean(MPE, 2), 2);
    %loss    = sum(RMS + MPE, 2) / size(AccPred, 2);
end