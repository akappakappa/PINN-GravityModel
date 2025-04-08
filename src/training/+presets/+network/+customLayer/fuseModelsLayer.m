classdef fuseModelsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties
        e
    end

    methods
        function layer = fuseModelsLayer(args)
            arguments
                args.Name        = "fuseModelsLayer";
                args.Description = "Fuses the Neural Network and Low-Fidelity Analytic Model Potentials";
                args.InputNames  = ["PotNN", "PotLF", "Radius"];
                args.OutputNames = "Potential";
                args.e
            end
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            layer.e           = args.e;
        end

        function Potential = predict(layer, PotNN, PotLF, Radius)
            refFusion         = 1 + layer.e;
            smoothFusion      = 0.5;                                                     % Slower transition
            weightLowFidelity = (1 + tanh(smoothFusion .* (Radius - refFusion))) ./ 2;   % Smooth transition from Network to Low-Fidelity model around 1+e
            Potential         = PotNN + weightLowFidelity .* PotLF;
        end
    end
end