function Potential = scaleNNPotentialLayer(Potential, Radius)
    ScaleFactor                   = Radius;
    ScaleFactor(ScaleFactor <= 1) = 1;
    Potential                     = Potential ./ ScaleFactor;
end