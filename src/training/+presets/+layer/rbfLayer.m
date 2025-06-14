classdef rbfLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % rbfLayer Radial Basis Function Layer
    %   This layer implements a radial basis function (RBF) activation function.
    
    properties (Learnable)
        Centers   % Centers of the RBFs
        Gamma     % Inverse Sigma^2
    end

    properties
        InputSize
        OutputSize
    end
    
    methods
        function layer = rbfLayer(InputSize, OutputSize, args)
            arguments
                InputSize
                OutputSize
                args.Name        = "rbfLayer";
                args.Description = "Radial Basis Function Layer";
            end
            
            layer.Name        = args.Name;
            layer.InputSize   = InputSize;
            layer.OutputSize  = OutputSize;
            layer.Description = args.Description;

            % RBF
            layer.Centers = dlarray(rand(InputSize, OutputSize));
            layer.Gamma   = dlarray(ones(OutputSize, 1));
        end
        
        function Z = predict(layer, X)
            % Computes the output of the layer using the RBF activation function.
            %cSize = size(layer.Centers, 2);
            %Dists = dlarray(zeros(cSize, size(X, 2)), "CB");
            %for i = 1:cSize
            %    Dists(i, :) = sum((layer.Centers(:, i) - X) .^ 2) .^ 0.5 .* layer.Bias(i);
            %end
            %Z = exp(-Dists .^2);

            % ||x-c||^2 = ||x||^2 + ||c||^2 - 2(x*c)
            C2    = sum(layer.Centers .^ 2, 1);
            X2    = sum(stripdims(X) .^ 2, 1);
            XC    = stripdims(X)' * layer.Centers;
            Dists = dlarray(((X2' + C2) - 2 * XC)', "CB");

            Z = exp(-layer.Gamma .* Dists);
        end
    end
end