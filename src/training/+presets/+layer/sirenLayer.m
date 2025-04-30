classdef sirenLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties (Learnable)
        Weights
        Bias
    end
    
    properties
        Omega0
        Dummy
    end
    
    methods
        function layer = sirenLayer(InputSize, OutputSize, Omega0, args)
            arguments
                InputSize
                OutputSize
                Omega0      = 30;
                args.Name        = "sirenLayer";
                args.Description = "SIREN Layer";
            end
            layer.Name   = args.Name;
            layer.Omega0 = Omega0;

            % Weight and bias initialization
            limit         = sqrt(6 / (InputSize * Omega0 ^ 2));
            weights       = (rand(OutputSize, InputSize) * 2 - 1) * limit;
            layer.Weights = dlarray(weights);
            layer.Bias    = dlarray(zeros(OutputSize, 1));
            layer.Dummy   = dlarray(zeros(OutputSize, 1));
        end
        
        function Z = predict(layer, X)
            Z_linear = fullyconnect(X, layer.Weights, layer.Dummy);
            Z = sin(layer.Omega0 * (Z_linear) + layer.Bias);
        end
        
    end
end
