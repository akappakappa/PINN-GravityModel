classdef analyticModelLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % analyticModelLayer Compute a Low-Fidelity Analytic Model for the Potential.
    %   This layer computes a low-fidelity analytic model based on the 'mu' physical parameter.

    properties
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
            layer.rref        = 0;
            layer.smoothness  = 0.5;
        end

        function Potential = predict(layer, Radius)
            % Computes the potential at the given RADIUS.

            Potential = -(layer.mu ./ Radius);

            % Tanh
            weight    = (1 + tanh(layer.smoothness .* (Radius - layer.rref))) ./ 2;

            % Smoothstep
            %weight                = zeros(size(Radius));
            %rStart                = layer.rref - layer.smoothness;
            %rEnd                  = layer.rref + layer.smoothness;
            %mask                  = Radius >= rStart & Radius <= rEnd;
            %x                     = (Radius(mask) - rStart) / (rEnd - rStart);
            %%weight(mask)          = x .^ 2 .* (3 - 2 .* x);
            %weight(mask)          = x .^ 3 .* (x .* (6 .* x - 15) + 10);
            %weight(Radius > rEnd) = 1;

            Potential = weight .* Potential;
        end
    end
end