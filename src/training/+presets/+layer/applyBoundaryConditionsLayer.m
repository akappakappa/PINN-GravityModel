classdef applyBoundaryConditionsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % applyBoundaryConditionsLayer Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model.
    %   This layer transitions predictions from the Fused Model to the Low-Fidelity Analytic Model using a smooth transition around 10R.

    properties
        rref        % Reference radius for the model
        smoothness  % Smoothness of the model transition
    end

    methods
        function layer = applyBoundaryConditionsLayer(args)
            arguments
                args.Name        = "applyBoundaryConditionsLayer";
                args.Description = "Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model";
                args.InputNames  = ["PotFused", "PotLF", "Radius"];
                args.OutputNames = "Potential";
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            layer.rref        = 14;
            layer.smoothness  = 4;
        end

        function Potential = predict(layer, PotFused, PotLF, Radius)
            % Computes the potential at the given RADIUS, applying a smooth transition around 10R between the Fused Model and the Low-Fidelity Analytic Model.

            % GELU
            %weightBounds     = (1 + erf(layer.smoothness .* (Radius - layer.rref))) ./ 2;

            % Tanh
            %weightBounds     = (1 + tanh(layer.smoothness .* (Radius - layer.rref))) ./ 2;

            % Sigmoid
            %weightBounds     = 1 ./ (1 + exp(-layer.smoothness .* (Radius - layer.rref)));

            % Ramp
            %weightBounds              = zeros(size(Radius));
            %idxLinear                 = Radius >= 8 & Radius <= 12;
            %weightBounds(idxLinear)   = (Radius(idxLinear) - 8) / 4;
            %weightBounds(Radius > 12) = 1;

            % Smoothstep
            weightBounds                = zeros(size(Radius));
            rStart                      = layer.rref - layer.smoothness;
            rEnd                        = layer.rref + layer.smoothness;
            mask                        = Radius >= rStart & Radius <= rEnd;
            x                           = (Radius(mask) - rStart) / (rEnd - rStart);
            weightBounds(mask)          = x .^ 2 .* (3 - 2 .* x);
            %weightBounds(mask)          = x .^ 3 .* (x .* (6 .* x - 15) + 10);
            weightBounds(Radius > rEnd) = 1;

            weightNetwork    = 1 - weightBounds;
            Potential        = weightNetwork .* PotFused + weightBounds .* PotLF;
        end
    end
end