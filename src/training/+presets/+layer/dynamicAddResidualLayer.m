classdef dynamicAddResidualLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    methods
        function layer = dynamicAddResidualLayer(args)
            arguments
                args.Name        = "dynamicAddResidualLayer";
                args.Description = "A residual addition layer that dynamically deactivates computation";
                args.InputNames  = ["Input", "Residual"];
                args.OutputNames = "Output";
            end
            % Construct the layer.
            
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Output = predict(~, Input, Residual)

            if isempty(Input) || isempty(Residual)
                Output = Input;
                return;
            end
            Output = Input + Residual;
        end
    end
end