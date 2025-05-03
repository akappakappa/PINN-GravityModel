classdef factorizedSirenLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties (Learnable)
        W1
        W2
        Bias
    end
    
    properties
        InputSize
        OutputSize
        Rank
        Omega0
    end
    
    methods
        function layer = factorizedSirenLayer(InputSize, OutputSize, Rank, Omega0, args)
            arguments
                InputSize
                OutputSize
                Rank
                Omega0
                args.Name        = "factorizedSirenLayer";
                args.Description = "Factorized Siren layer for fewer parameters";
            end
            layer.Name       = args.Name;
            layer.InputSize  = InputSize;
            layer.OutputSize = OutputSize;
            layer.Rank       = Rank;
            layer.Omega0     = Omega0;

            % Glorot initialization for full-rank matrix W
            bound = sqrt(6 / (InputSize * Omega0 ^ 2));
            Z     = 2 * rand([OutputSize, InputSize]) - 1;
            W     = bound * Z;

            % Spectral initialization for W1 and W2
            [U, S, V] = svds(W, Rank);
            layer.W1  = dlarray(U * sqrt(S));
            layer.W2  = dlarray(sqrt(S) * V');

            % Bias initialization
            layer.Bias = dlarray(zeros(OutputSize, 1));
        end
        
        function Z = predict(layer, X)
            W = layer.W1 * layer.W2;
            Z = sin(layer.Omega0 *fullyconnect(X, W, layer.Bias));
        end
    end
end