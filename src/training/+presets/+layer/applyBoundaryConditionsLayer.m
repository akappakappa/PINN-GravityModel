classdef applyBoundaryConditionsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % applyBoundaryConditionsLayer Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model.
    %   This layer transitions predictions from the Fused Model to the Low-Fidelity Analytic Model using a smooth transition around 10R.

    properties
        H           % Smooth transition function
    end

    properties (Learnable)
        rref        % Reference radius for the model
        smoothness  % Smoothness of the model transition
    end

    methods
        function layer = applyBoundaryConditionsLayer(args)
            arguments
                args.Name        = "applyBoundaryConditionsLayer";
                args.Description = "Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model";
                args.InputNames  = ["PotFused", "PotLF", "Radius"];
                args.OutputNames = "Potential";
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            layer.H           = @(x, k, r) (1 + tanh(k .* (x - r))) ./ 2;
            layer.rref        = 10;
            layer.smoothness  = 0.1;
        end

        function Potential = predict(layer, PotFused, PotLF, Radius)
            % Computes the potential at the given RADIUS, applying a smooth transition around 10R between the Fused Model and the Low-Fidelity Analytic Model.

            weightBounds  = layer.H(Radius, layer.smoothness, layer.rref);
            weightNetwork = 1 - weightBounds;
            Potential     = weightNetwork .* PotFused + weightBounds .* PotLF;
        end
    end
end