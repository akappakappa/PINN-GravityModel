classdef analyticModelLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % analyticModelLayer Compute a low-fidelity analytic gravitational potential model
    % The low-fidelity point-mass analytical model is meant to aid the Network's prediction.
    %
    % analyticModelLayer Properties:
    %    Mu         - Physical property of the celestial body defined as: Mu = G * Volume * Density
    %    Rref       - Reference radius to apply a gentle fade-in of this model's prediction
    %    Smoothness - Transition coefficient for the fade-in
    %
    % analyticModelLayer Methods:
    %    predict - Provide a low-fidelity gravitational potential prediction
    %
    % See also presets.layer.cart2sphLayer, presets.layer.fuseModelsLayer, presets.layer.applyBoundaryConditionsLayer.

    properties
        Mu
        Rref
        Smoothness
    end

    methods
        function layer = analyticModelLayer(Mu, args)
            arguments
                Mu

                args.Name        = "analyticModelLayer"
                args.InputNames  = ["Radius", "RadiusInvExt"]
                args.OutputNames = "Potential"

                args.Rref       = 0
                args.Smoothness = 0.5
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;

            layer.Mu         = Mu;
            layer.Rref       = args.Rref;
            layer.Smoothness = args.Smoothness;
        end

        function Potential = predict(layer, Radius, RadiusInvExt)
            Potential = -(layer.Mu .* RadiusInvExt);
            weight    = 0.5 + 0.5 .* tanh(layer.Smoothness .* (Radius - layer.Rref));
            Potential = weight .* Potential;
        end
    end
end