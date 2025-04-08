function Potential = applyBoundaryConditionsLayer(PotFused, PotLF, Radius)
    refBounds     = 10;                                                      % 10R = max altitude of the training dataset
    smoothBounds  = 2;                                                       % Faster transition
    weightBounds  = (1 + tanh(smoothBounds .* (Radius - refBounds))) ./ 2;   % Smooth transition from Network to Boundary Conditions around 10R
    weightNetwork = 1 - weightBounds;
    Potential     = weightNetwork .* PotFused + weightBounds .* PotLF;
end