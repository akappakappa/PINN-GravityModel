classdef applyBoundaryConditionsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    methods
        function layer = applyBoundaryConditionsLayer(args)
            arguments
                args.Name        = "applyBoundaryConditionsLayer";
                args.Description = "Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model";
                args.InputNames  = ["PotFused", "PotLF", "Radius"];
                args.OutputNames = "Potential";
            end
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Potential = predict(~, PotFused, PotLF, Radius)
            refBounds     = 10;                                                      % 10R = max altitude of the training dataset
            smoothBounds  = 2;                                                       % Faster transition
            weightBounds  = (1 + tanh(smoothBounds .* (Radius - refBounds))) ./ 2;   % Smooth transition from Network to Boundary Conditions around 10R
            weightNetwork = 1 - weightBounds;
            Potential     = weightNetwork .* PotFused + weightBounds .* PotLF;
        end
    end
end