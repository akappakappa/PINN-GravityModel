classdef sirenLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % sirenLayer SIREN Layer
    %   This layer implements the SIREN activation function, which is a sine activation function with learnable weights and bias.

    properties (Learnable)
        Weights   % Weights matrix
        Bias      % Bias vector
    end
    
    properties
        InputSize    % Size of the input layer
        OutputSize   % Size of the output layer
        Omega0       % Frequency parameter
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
            % Construct the layer, performing Glorot initialization for weights and setting the frequency parameter Omega0.

            layer.Name       = args.Name;
            layer.InputSize  = InputSize;
            layer.OutputSize = OutputSize;
            layer.Omega0     = Omega0;
        end

        function layer = initialize(layer, varargin)
            % Glorot initialization
            bound = sqrt(6 / (layer.InputSize * layer.Omega0 ^ 2));
            Z     = 2 * rand([layer.OutputSize, layer.InputSize]) - 1;
            W     = layer.Omega0 * bound * Z;

            % Layer properties
            layer.Weights = dlarray(W);
            layer.Bias    = dlarray(zeros(layer.OutputSize, 1));
        end
        
        function Z = predict(layer, X)
            % Computes the output of the layer using the weights and sine activation function.

            Z = sin(layer.Omega0 * fullyconnect(X, layer.Weights, layer.Bias));
        end
    end
end