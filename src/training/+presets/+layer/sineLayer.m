classdef sineLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % sineLayer Sine activation function
    % Sine activation function with Omega0 parameter (SIREN)
    %
    % sineLayer Properties:
    %    Omega0 - Scaling factor before sin() activation
    %
    % sineLayer Methods:
    %    predict - Applies Omega0 scaling to input X, then performs sin() activation
    
    properties
        Omega0
    end

    methods
        function layer = sineLayer(args)
            arguments
                args.Name   = "sineLayer"
                args.Omega0 = 1;
            end
            
            layer.Name = args.Name;

            layer.Omega0 = args.Omega0;
        end
        
        function Z = predict(layer, X)
            Z = sin(layer.Omega0 .* X);
        end
    end
end