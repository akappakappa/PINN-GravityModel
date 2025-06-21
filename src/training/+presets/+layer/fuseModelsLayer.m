classdef fuseModelsLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % fuseModelsLayer Fuse the Network's prediction with a low-fidelity analytic model
    % Samples that lie withing the training bounds are provided with a starting point prediction based on a low-fidelity analytic model.
    % This allows the Network to solely focus on learning the difference between a simple point-mass model and the complex true labels.
    %
    % fuseModelsLayer Methods:
    %    predict - Sum the Network's prediction to the low-fidelity model
    %
    % See also presets.layer.analyticModelLayer, presets.layer.applyBoundaryConditionsLayer.

    methods
        function layer = fuseModelsLayer(args)
            arguments
                args.Name        = "fuseModelsLayer"
                args.InputNames  = ["PotNN", "PotLF"]
                args.OutputNames = "Potential"
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Potential = predict(~, PotNN, PotLF)
            Potential = PotNN + PotLF;
        end
    end
end