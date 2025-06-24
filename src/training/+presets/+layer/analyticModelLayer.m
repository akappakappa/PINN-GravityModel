classdef analyticModelLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % analyticModelLayer Compute a low-fidelity analytic gravitational potential model
    % The low-fidelity point-mass analytical model is meant to aid the Network's prediction.
    %
    % analyticModelLayer Properties:
    %    Mu         - Physical property of the celestial body defined as: Mu = G * Volume * Density
    %    Rref       - Reference radius to apply a gentle fade-in of this model's prediction
    %    Smoothness - Transition coefficient for the fade-in
    %    WeightFunc - Transition function
    %
    % analyticModelLayer Methods:
    %    predict - Provide a low-fidelity gravitational potential prediction
    %
    % See also presets.layer.cart2sphLayer, presets.layer.fuseModelsLayer, presets.layer.applyBoundaryConditionsLayer.

    properties
        Mu
        Rref
        Smoothness
        WeightFunc
    end

    methods
        function layer = analyticModelLayer(Mu, args)
            arguments
                Mu

                args.Name        = "analyticModelLayer"
                args.InputNames  = "Radius"
                args.OutputNames = "Potential"

                args.Rref       = 0
                args.Smoothness = 0.5
                args.FadeIn     = true
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;

            layer.Mu         = Mu;
            layer.Rref       = args.Rref;
            layer.Smoothness = args.Smoothness;

            if (true == args.FadeIn)
                layer.WeightFunc = @layer.weightTanh;
            else
                layer.WeightFunc = @layer.weightOnes;
            end
        end

        function Potential = predict(layer, Radius)
            Potential = -(layer.Mu ./ Radius);
            weight    = layer.WeightFunc(Radius);
            Potential = weight .* Potential;
        end
    end

    methods (Access = private)
        function W = weightTanh(layer, Radius)
            W = 0.5 + 0.5 .* tanh(layer.Smoothness .* (Radius - layer.Rref));
        end

        function W = weightOnes(~, Radius)
            W = ones(size(Radius));
        end
    end
end