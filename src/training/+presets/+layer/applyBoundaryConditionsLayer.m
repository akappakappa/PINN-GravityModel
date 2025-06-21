classdef applyBoundaryConditionsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % applyBoundaryConditionsLayer Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model.
    %   This layer transitions predictions from the Fused Model to the Low-Fidelity Analytic Model using a smooth transition around 10R.

    properties
        rref        % Reference radius for the model
        smoothness  % Smoothness of the model transition
        weightFunc
    end

    methods
        function layer = applyBoundaryConditionsLayer(args)
            arguments
                args.Name        = "applyBoundaryConditionsLayer";
                args.Description = "Applies Boundary Conditions to transition from the Fused Model to the Low-Fidelity Analytic Model";
                args.InputNames  = ["PotFused", "PotLF", "Radius"];
                args.OutputNames = "Potential";
                args.Mode {mustBeMember(args.Mode, {'smoothstep', 'tanh'})} = "smoothstep";
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            
            switch args.Mode
                case "tanh"
                    layer.rref       = 10;
                    layer.smoothness = 0.5;
                    layer.weightFunc = @layer.weightTanh;
                case "smoothstep"
                    layer.rref       = 14;
                    layer.smoothness = 4;
                    layer.weightFunc = @layer.weightSmoothstep;
            end
        end

        function Potential = predict(layer, PotFused, PotLF, Radius)
            % Computes the potential at the given RADIUS, applying a smooth transition around 10R between the Fused Model and the Low-Fidelity Analytic Model.

            weightBounds  = layer.weightFunc(Radius);
            weightNetwork = 1 - weightBounds;
            Potential     = weightNetwork .* PotFused + weightBounds .* PotLF;
        end


        
        function W = weightSmoothstep(layer, Radius)
            % Smoothstep polynomial function

            R1   = layer.rref - layer.smoothness;
            R2   = layer.rref + layer.smoothness;
            mask = Radius >= R1 & Radius <= R2;
            x    = (Radius(mask) - R1) / (R2 - R1);

            W            = zeros(size(Radius));
            W(mask)      = x .^ 2 .* (3 - 2 .* x);
            W(mask > R2) = 1;
        end

        function W = weightTanh(layer, Radius)
            % Tanh based fuction, prone to artefacting

            W = (1 + tanh(layer.smoothness .* (Radius - layer.rref))) ./ 2;
        end
    end
end