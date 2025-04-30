classdef factorizedLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties (Learnable)
        W1
        W2
        Bias
    end
    
    properties
        InputSize
        OutputSize
        Rank
        Dummy
    end
    
    methods
        function layer = factorizedLayer(args)
            arguments
                args.Name        = "factorizedLayer";
                args.Description = "Factorized layer for fewer parameters";
                args.InputSize
                args.OutputSize
                args.Rank
            end
            layer.Name       = args.Name;
            layer.InputSize  = args.InputSize;
            layer.OutputSize = args.OutputSize;
            layer.Rank       = args.Rank;

            % Initialized weights
            limit1      = sqrt(6 / (args.InputSize + args.Rank));
            limit2      = sqrt(6 / (args.Rank + args.OutputSize));
            layer.W1    = dlarray((rand(args.Rank, args.InputSize) * 2 - 1) * limit1);
            layer.W2    = dlarray((rand(args.OutputSize, args.Rank) * 2 - 1) * limit2);
            layer.Bias  = dlarray(zeros(args.OutputSize, 1));
            layer.Dummy = dlarray(zeros(args.Rank, 1));
        end
        
        function Z = predict(layer, X)
            Z = fullyconnect(fullyconnect(X, layer.W1, layer.Dummy), layer.W2, layer.Bias);
        end
        
    end
end
