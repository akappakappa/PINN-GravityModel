classdef scaleNNPotentialLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % scaleNNPotentialLayer Scale Network's prediction by radius-based scale factor
    % Scaling the network prediction by 1/r allows the Network to learn an altitude-independent gravitational potential.
    % This greatly increases numerical stability.
    %
    % scaleNNPotentialLayer Properties:
    %    AnalyticModelPower - Power value of the low-fidelity analytic model, decreases this (Network's) model power
    %
    % scaleNNPotentialLayer Methods:
    %    predict - Performs 1/r scaling of Network's learned potential
    %
    % See also presets.layer.cart2sphLayer.

    properties
        AnalyticModelPower
    end

    methods
        function layer = scaleNNPotentialLayer(args)
            arguments
                args.Name        = "scaleNNPotentialLayer"
                args.InputNames  = ["Potential", "Radius"]
                args.OutputNames = "Potential"

                args.AnalyticModelPower = 1
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;

            layer.AnalyticModelPower = args.AnalyticModelPower;
        end

        function Potential = predict(layer, Potential, Radius)
            Potential = Potential ./ (max(Radius, 1) .^ layer.AnalyticModelPower);
        end
    end
end