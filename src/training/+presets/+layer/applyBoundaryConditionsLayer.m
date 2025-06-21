classdef applyBoundaryConditionsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % applyBoundaryConditionsLayer Transition from fused (Network + low-fidelity) potential to exclusively low-fidelity prediction.
    % Samples that lie outside the training bounds will use a low-fidelity potential to avoid high generalization error numbers at high altitudes.
    % Construct with either "smoothstep" (default, ideal) or "tanh" (prone to artefacting) Mode argument.
    %
    % applyBoundaryConditionsLayer Properties:
    %    Rref       - Reference radius to apply a gentle fade-in of this model's prediction
    %    Smoothness - Transition coefficient for the fade-in
    %    WeightFunc - Transition function
    %
    % applyBoundaryConditionsLayer Methods:
    %    predict - Transition from exterior bounds prediction to extrapolation prediction
    %
    % See also presets.layer.cart2sphLayer, presets.layer.analyticModelLayer, presets.layer.fuseModelsLayer.
    
    properties
        Rref
        Smoothness
        WeightFunc
    end

    methods
        function layer = applyBoundaryConditionsLayer(args)
            arguments
                args.Name        = "applyBoundaryConditionsLayer"
                args.InputNames  = ["PotFused", "PotLF", "Radius"]
                args.OutputNames = "Potential"

                args.Mode {mustBeMember(args.Mode, {'smoothstep', 'tanh'})} = "smoothstep";
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            
            switch args.Mode
                case "tanh"
                    layer.Rref       = 10;
                    layer.Smoothness = 0.5;
                    layer.WeightFunc = @layer.weightTanh;
                case "smoothstep"
                    layer.Rref       = 14;
                    layer.Smoothness = 4;
                    layer.WeightFunc = @layer.weightSmoothstep;
            end
        end

        function Potential = predict(layer, PotFused, PotLF, Radius)
            weightBounds  = layer.WeightFunc(Radius);
            weightNetwork = 1 - weightBounds;
            Potential     = weightNetwork .* PotFused + weightBounds .* PotLF;
        end
    end

    methods (Access = private)
        function W = weightSmoothstep(layer, Radius)
            R1   = layer.Rref - layer.Smoothness;
            R2   = layer.Rref + layer.Smoothness;
            mask = Radius >= R1 & Radius <= R2;
            x    = (Radius(mask) - R1) / (R2 - R1);

            W            = zeros(size(Radius));
            W(mask)      = x .^ 2 .* (3 - 2 .* x);
            W(mask > R2) = 1;
        end

        function W = weightTanh(layer, Radius)
            W = (1 + tanh(layer.Smoothness .* (Radius - layer.Rref))) ./ 2;
        end
    end
end