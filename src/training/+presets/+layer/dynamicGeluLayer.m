classdef dynamicGeluLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    methods
        function layer = dynamicGeluLayer(args)
            arguments
                args.Name        = "dynamicGeluLayer";
                args.Description = "A Gaussian Error Linear Unit (GELU) activation function layer that dynamically deactivates computation";
                args.InputNames  = "Input";
                args.OutputNames = "Output";
            end
            % Construct the layer.
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Output = predict(~, Input)

            if isempty(Input)
                Output = Input;
                return;
            end
            Output = gelu(Input);
        end
    end
end