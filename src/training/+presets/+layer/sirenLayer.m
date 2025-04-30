classdef sirenLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties (Learnable)
        Weights
        Bias
    end
    
    properties
        Omega0
    end
    
    methods
        function layer = sirenLayer(InputSize, OutputSize, Omega0, args)
            arguments
                InputSize
                OutputSize
                Omega0
                args.Name        = "sirenLayer";
                args.Description = "SIREN Layer";
            end
            layer.Name   = args.Name;
            layer.Omega0 = Omega0;

            % Glorot initialization
            bound = sqrt(6 / (InputSize * Omega0 ^ 2));
            Z     = 2 * rand([OutputSize, InputSize]) - 1;
            W     = Omega0 * bound * Z;

            % Layer properties
            layer.Weights = dlarray(W);
            layer.Bias    = dlarray(zeros(OutputSize, 1));
        end
        
        function Z = predict(layer, X)
            Z = sin(layer.Omega0 * fullyconnect(X, layer.Weights, layer.Bias));
        end
    end
end