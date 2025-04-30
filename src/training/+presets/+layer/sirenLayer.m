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
        function layer = sirenLayer(inputSize, outputSize, omega0, name)
            layer.Name = name;
            layer.Description = "SIREN Layer with Omega0 = " + omega0;
            layer.Omega0 = omega0;

            % Weight and bias initialization
            limit = sqrt(6 / inputSize) / omega0;
            weights = (rand(outputSize, inputSize) * 2 - 1) * limit;
            layer.Weights = dlarray(weights);
            layer.Bias = dlarray(zeros(outputSize, 1));
            layer.Dummy = dlarray(zeros(outputSize, 1));
        end
        
        function Z = predict(layer, X)
            Z_linear = fullyconnect(X, layer.Weights, layer.Dummy);
            Z = sin(layer.Omega0 * (Z_linear) + layer.Bias);
        end
        
    end
end
