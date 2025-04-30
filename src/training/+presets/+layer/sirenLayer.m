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
        function layer = sirenLayer(args)
            arguments
                args.Name        = "sirenLayer";
                args.Description = "SIREN Layer";
                args.InputSize
                args.OutputSize
                args.Omega0      = 30;
            end
            layer.Name   = args.Name;
            layer.Omega0 = args.Omega0;

            % Weight and bias initialization
            limit         = sqrt(6 / args.InputSize) / args.Omega0;
            weights       = (rand(args.OutputSize, args.InputSize) * 2 - 1) * limit;
            layer.Weights = dlarray(weights);
            layer.Bias    = dlarray(zeros(args.OutputSize, 1));
            layer.Dummy   = dlarray(zeros(args.OutputSize, 1));
        end
        
        function Z = predict(layer, X)
            Z_linear = fullyconnect(X, layer.Weights, layer.Dummy);
            Z = sin(layer.Omega0 * (Z_linear) + layer.Bias);
        end
        
    end
end
