classdef filmLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable

    properties (Learnable)
        Scale
        Shift
    end

    methods
        function layer = filmLayer(NumFeatures, args)
            arguments
                NumFeatures
                args.Name        = "filmLayer";
                args.Description = "Feature Modulation layer"
            end

            layer.Name        = args.Name;
            layer.Description = args.Description;

            layer.Scale = dlarray(ones(NumFeatures, 1));
            layer.Shift = dlarray(zeros(NumFeatures, 1));
        end

        function Z = predict(layer, X)

            Z = layer.Scale .* X + layer.Shift;
        end
    end
end