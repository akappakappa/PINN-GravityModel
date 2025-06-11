classdef mergingLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % mergingLayer Compute a Low-Fidelity Analytic Model for the Potential, then merge it with the Neural Network Potential.
    %   This layer computes a low-fidelity analytic model based on the 'mu' physical parameter, then merges it with the Neural Network Potential.

    properties
        r1          % Reference radius for the model
        r2          % Second reference radius for the model
        smoothness  % Smoothness of the model transition
        mu          % Physical parameter
    end

    methods
        function layer = mergingLayer(args)
            arguments
                args.Name        = "mergingLayer";
                args.Description = "Computes a Low-Fidelity Analytic Model for the Potential";
                args.InputNames  = ["Potential", "Radius"];
                args.OutputNames = "Potential";
                args.mu
            end
            % Construct the layer, given MU.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            layer.mu          = args.mu;
            layer.r1          = 0;
            layer.r2          = 10;
            layer.smoothness  = 0.5;
        end

        function Potential = predict(layer, Potential, Radius)
            % Computes the potential at the given RADIUS, using a low-fidelity analytic model based on the 'mu' physical parameter, then merges it with the Neural Network Potential.
            PotLF    = -(layer.mu ./ Radius);
            weightLF = (1 + tanh(layer.smoothness .* (Radius - layer.r1))) ./ 2;

            %wgtInt   = 1 - (1 + tanh(layer.smoothness .* (Radius - layer.r2))) ./ 2;
            %wgtExt   = 1 - max(0, min(1, (Radius - (layer.r2 - 2)) ./ 4));
            %maskInt  = Radius < layer.r2;
            %maskExt  = 1 - maskInt;
            %weightNN = maskInt .* wgtInt + maskExt .* wgtExt;


            weightNN = ...
                (Radius < layer.r2)  .* (1 - (1 + tanh(layer.smoothness .* (Radius - layer.r2))) ./ 2) + ...
                (Radius >= layer.r2) .* (1 - (1 + tanh(layer.smoothness .* ((Radius - layer.r2) + 0.2 .* (Radius - layer.r2) .^ 3))) ./ 2);

            Potential = weightNN .* Potential + weightLF .* PotLF;
        end
    end
end