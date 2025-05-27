classdef analyticModelLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % analyticModelLayer Compute a Low-Fidelity Analytic Model for the Potential.
    %   This layer computes a low-fidelity analytic model based on the 'mu' physical parameter.

    properties
        H           % Smooth transition function
        rref        % Reference radius for the model
        smoothness  % Smoothness of the model transition
        mu          % Physical parameter
    end

    methods
        function layer = analyticModelLayer(args)
            arguments
                args.Name        = "analyticModelLayer";
                args.Description = "Computes a Low-Fidelity Analytic Model for the Potential";
                args.InputNames  = "Radius";
                args.OutputNames = "Potential";
                args.mu
            end
            % Construct the layer, given MU.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            layer.mu          = args.mu;
            layer.H           = @(x, k, r) (1 + tanh(k .* (x - r))) ./ 2;
            layer.rref        = 0;
            layer.smoothness  = 0.5;
        end

        function Potential = predict(layer, Radius)
            % Computes the potential at the given RADIUS.

            Potential = -(layer.mu ./ Radius);
            Potential = layer.H(Radius, layer.smoothness, layer.rref) .* Potential;
        end
    end
end