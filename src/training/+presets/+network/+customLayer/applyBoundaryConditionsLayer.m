function Potential = applyBoundaryConditionsLayer(PotFused, PotLF, Radius)
    function weight = Transition(radius, reference, smoothing)
        weight = (1 + tanh(smoothing * (radius - reference))) / 2;
    end
    refBounds     = 10;                                           % 10R = max altitude of the training dataset
    smoothBounds  = 2;                                            % Faster transition
    weightBounds  = Transition(Radius, refBounds, smoothBounds);   % Smooth transition from Network to Boundary Conditions around 10R
    weightNetwork = 1 - weightBounds;
    
    Potential = weightNetwork .* PotFused + weightBounds .* PotLF;
end