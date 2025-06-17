classdef gateLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable

    properties (Learnable)
        Gate
    end

    methods
        function layer = gateLayer(NumFeatures, args)
            arguments
                NumFeatures
                args.Name        = "gateLayer";
                args.Description = "Gating layer"
            end

            layer.Name        = args.Name;
            layer.Description = args.Description;

            layer.Gate = dlarray(zeros(NumFeatures, 1));
        end

        function Z = predict(layer, X)

            Z = sigmoid(layer.Gate) .* X;
        end
    end
end