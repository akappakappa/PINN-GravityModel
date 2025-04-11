classdef analyticModelLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties
        mu
    end

    methods
        function layer = analyticModelLayer(args)
            arguments
                args.Name        = "analyticModelLayer";
                args.Description = "Computes a Low-Fidelity Analytic Model for the Potential";
                args.InputNames  = "Radius";
                args.OutputNames = "Potential";
                args.mu
            end
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            layer.mu          = args.mu;
        end

        function Potential = predict(layer, Radius)
            fx        = 0;
            Potential = -(layer.mu ./ Radius + fx);
        end
    end
end