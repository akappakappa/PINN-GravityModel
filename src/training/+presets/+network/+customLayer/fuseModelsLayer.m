function Potential = fuseModelsLayer(PotNN, PotLF, Radius, e)
    function weight = Transition(radius, reference, smoothing)
        weight = (1 + tanh(smoothing * (radius - reference))) / 2;
    end
    refFusion         = 1 + e;
    smoothFusion      = 0.5;                                          % Slower transition
    weightLowFidelity = Transition(Radius, refFusion, smoothFusion);   % Low-Fidelity model gradually more important around 1+e
    Potential         = PotNN + weightLowFidelity .* PotLF;
end