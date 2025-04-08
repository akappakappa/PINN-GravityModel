function Potential = scaleNNPotentialLayer(Potential, Radius)
    ScaleFactor = max(Radius, 1);
    Potential   = Potential ./ ScaleFactor;
end