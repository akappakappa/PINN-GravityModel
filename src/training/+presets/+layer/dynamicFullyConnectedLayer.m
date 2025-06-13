classdef dynamicFullyConnectedLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties (Learnable)
        Weights
        Bias
    end

    properties
        InputSize
        OutputSize
    end

    methods
        function layer = dynamicFullyConnectedLayer(InputSize, OutputSize, args)
            arguments
                InputSize
                OutputSize
                args.Name        = "dynamicFullyConnectedLayer";
                args.Description = "A fully connected layer that dynamically deactivates computation";
                args.WeightsInit = "glorot";
            end
            % Construct the layer.
            layer.Name        = args.Name;
            layer.InputSize   = InputSize;
            layer.OutputSize  = OutputSize;
            layer.Description = args.Description;

            switch args.WeightsInit
                case "glorot"
                    bound         = sqrt(6 / (InputSize + OutputSize));
                    Z             = 2 * rand([OutputSize, InputSize]) - 1;
                    layer.Weights = dlarray(bound * Z);
                case "zeros"
                    layer.Weights = dlarray(zeros(OutputSize, InputSize));
                otherwise
                    error("Unsupported weight initialization method: %s", args.WeightsInit);
            end
            layer.Bias = dlarray(zeros(OutputSize, 1));
        end

        function Z = predict(layer, X)

            if isempty(X)
                Z = dlarray(zeros(layer.OutputSize, 0), "CB");
                return;
            end
            Z = fullyconnect(X, layer.Weights, layer.Bias);
        end
    end
end