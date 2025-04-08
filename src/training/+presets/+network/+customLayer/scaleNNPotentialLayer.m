classdef scaleNNPotentialLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    methods
        function layer = scaleNNPotentialLayer(args)
            arguments
                args.Name        = "scaleNNPotentialLayer";
                args.Description = "Scales the learned proxy Potential to the physically correct one";
                args.InputNames  = ["Potential", "Radius"];
                args.OutputNames = "Potential";
            end
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Potential = predict(~, Potential, Radius)
            ScaleFactor = max(Radius, 1);
            Potential   = Potential ./ ScaleFactor;
        end
    end
end