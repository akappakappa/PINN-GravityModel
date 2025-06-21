classdef scaleNNPotentialLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % scaleNNPotentialLayer Scale Network's prediction by radius-based scale factor
    % Scaling the network prediction by 1/r allows the Network to learn an altitude-independent gravitational potential.
    % This greatly increases numerical stability.
    %
    % scaleNNPotentialLayer Methods:
    %    predict - Performs 1/r scaling of Network's learned potential
    %
    % See also presets.layer.cart2sphLayer.

    methods
        function layer = scaleNNPotentialLayer(args)
            arguments
                args.Name        = "scaleNNPotentialLayer"
                args.InputNames  = ["Potential", "Radius"]
                args.OutputNames = "Potential"
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Potential = predict(~, Potential, Radius)
            Potential = Potential ./ max(Radius, 1);
        end
    end
end