classdef applyBoundaryConditionsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % applyBoundaryConditionsLayer Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model.
    %   This layer transitions predictions from the Fused Model to the Low-Fidelity Analytic Model using a smooth transition around 10R.

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
        end

        function Potential = predict(~, PotFused, PotLF, Radius)
            % Computes the potential at the given RADIUS, applying a smooth transition around 10R between the Fused Model and the Low-Fidelity Analytic Model.

            refBounds     = 10;                                                      % 10R = max altitude of the training dataset
            smoothBounds  = 0.5;                                                     % Transition
            weightBounds  = (1 + tanh(smoothBounds .* (Radius - refBounds))) ./ 2;   % Smooth transition from Network to Boundary Conditions around 10R
            weightNetwork = 1 - weightBounds;
            Potential     = weightNetwork .* PotFused + weightBounds .* PotLF;
        end
    end
end