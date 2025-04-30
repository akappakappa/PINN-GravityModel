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
        function layer = factorizedLayer(inputSize, outputSize, rank, name)
            layer.Name = name;
            layer.InputSize = inputSize;
            layer.OutputSize = outputSize;
            layer.Rank = rank;

            % Initialized weights
            limit1 = sqrt(6 / (inputSize + rank));
            limit2 = sqrt(6 / (rank + outputSize));
            layer.W1 = dlarray((rand(rank, inputSize) * 2 - 1) * limit1);
            layer.W2 = dlarray((rand(outputSize, rank) * 2 - 1) * limit2);
            layer.Bias = dlarray(zeros(outputSize, 1));
            layer.Dummy = dlarray(zeros(rank, 1));
        end
        
        function Z = predict(layer, X)
            Z = fullyconnect(fullyconnect(X, layer.W1, layer.Dummy), layer.W2, layer.Bias);
        end
        
    end
end
