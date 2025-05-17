classdef analyticModelLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % analyticModelLayer Compute a Low-Fidelity Analytic Model for the Potential.
    %   This layer computes a low-fidelity analytic model based on the 'mu' physical parameter.

    properties
        mu   % Physical parameter
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
        end

        function Potential = predict(layer, Radius)
            % Computes the potential at the given RADIUS, differentiating between internal (0R:1R) and external (>1R) regions.

            uInternal = layer.mu .* Radius .^ 2 + 2 * (-layer.mu);
            uExternal = layer.mu ./ Radius;
            mask      = Radius < 1;
            Potential = mask .* uInternal + (1 - mask) .* uExternal;
        end
    end
end