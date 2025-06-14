classdef sineLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % sineLayer Sine Layer
    %   This layer implements the sine activation function.

    properties
        Omega0
    end

    methods
        function layer = sineLayer(args)
            arguments
                args.Name        = "sineLayer";
                args.Description = "Sine activation layer";
                args.Omega0      = 30;
            end
            
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.Omega0      = Omega0;
        end
        
        function Z = predict(layer, X)
            % Computes the output of the layer using the sine activation function.

            Z = sin(layer.Omega0 .* X);
        end
    end
end