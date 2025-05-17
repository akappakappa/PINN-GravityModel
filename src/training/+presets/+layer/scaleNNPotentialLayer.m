classdef scaleNNPotentialLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % scaleNNPotentialLayer Scales the learned proxy Potential to the physically correct one.
    %   This layer scales the learned radius-independent proxy Potential to the physically correct one, using the previously computed radius.

    methods
        function layer = scaleNNPotentialLayer(args)
            arguments
                args.Name        = "scaleNNPotentialLayer";
                args.Description = "Scales the learned proxy Potential to the physically correct one";
                args.InputNames  = ["Potential", "Radius"];
                args.OutputNames = "Potential";
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Potential = predict(~, Potential, Radius)
            % Computes the potential at the given RADIUS, scaling the learned proxy Potential to the physically correct one.

            ScaleFactor = max(Radius, 1);
            Potential   = Potential ./ ScaleFactor;
        end
    end
end