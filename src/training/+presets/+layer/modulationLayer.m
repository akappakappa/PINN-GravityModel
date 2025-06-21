classdef modulationLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % modulationLayer Learnable input modulation
    % Learnable scaling factor and shifting vector.
    %
    % gateLayer Properties:
    %    Scale - Learnable scale
    %    Shift - Learnable shift
    %
    % gateLayer Methods:
    %    initialize - Ones Scale weights initialization, Zeros Shift weights initialization
    %    predict    - Applies scaling and shifting to input

    properties (Learnable)
        Scale
        Shift
    end

    methods
        function layer = modulationLayer(args)
            arguments
                args.Name  = "modulationLayer"
                args.Scale = []
                args.Shift = []
            end

            layer.Name = args.Name;

            layer.Scale = args.Scale;
            layer.Shift = args.Shift;
        end

        function layer = initialize(layer, layout)
            numChannels = layout.Size(finddim(layout, "C"));

            if isempty(layer.Scale)
                layer.Scale = dlarray(ones(numChannels, 1));
            end

            if isempty(layer.Shift)
                layer.Shift = dlarray(zeros(numChannels, 1));
            end
        end

        function Z = predict(layer, X)
            Z = layer.Scale .* X + layer.Shift;
        end
    end
end