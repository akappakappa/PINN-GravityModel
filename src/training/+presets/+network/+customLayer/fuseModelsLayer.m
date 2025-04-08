function Potential = fuseModelsLayer(PotNN, PotLF, Radius, e)
    refFusion         = 1 + e;
    smoothFusion      = 0.5;                                                     % Slower transition
    weightLowFidelity = (1 + tanh(smoothFusion .* (Radius - refFusion))) ./ 2;   % Smooth transition from Network to Low-Fidelity model around 1+e
    Potential         = PotNN + weightLowFidelity .* PotLF;
end