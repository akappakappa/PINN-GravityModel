classdef fuseModelsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % fuseModelsLayer Fuses the Neural Network and Low-Fidelity Analytic Model Potentials.
    %   This layer fuses the Neural Network and Low-Fidelity Analytic Model Potentials.

    methods
        function layer = fuseModelsLayer(args)
            arguments
                args.Name        = "fuseModelsLayer";
                args.Description = "Fuses the Neural Network and Low-Fidelity Analytic Model Potentials";
                args.InputNames  = ["PotNN", "PotLF", "Radius"];
                args.OutputNames = "Potential";
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Potential = predict(~, PotNN, PotLF)
            % Computes the potential by fusing the Neural Network and Low-Fidelity Analytic Model Potentials.

            Potential = PotNN + PotLF;
        end
    end
end