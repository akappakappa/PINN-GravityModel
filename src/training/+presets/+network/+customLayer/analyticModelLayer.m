function Potential = analyticModelLayer(Radius, mu)
    fx        = 0;
    Potential = -(mu ./ Radius + fx);
end