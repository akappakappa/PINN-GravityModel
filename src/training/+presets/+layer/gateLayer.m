classdef gateLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % gateLayer Gate prediction with learnable sigmoid-ed vector
    % Learnable [0:1] gate.
    %
    % gateLayer Properties:
    %    Gate - Learnable gate
    %
    % gateLayer Methods:
    %    initialize - Zeros Gate weights initialization
    %    predict    - Applies sigmoid to learned Gate to keep it in [0:1], then multiplies it byt the input

    properties (Learnable)
        Gate
    end

    methods
        function layer = gateLayer(args)
            arguments
                args.Name = "gateLayer"
                args.Gate = []
            end

            layer.Name = args.Name;

            layer.Gate = args.Gate;
        end

        function layer = initialize(layer, layout)
            numChannels = layer.Size(finddim(layout, "C"));

            if isempty(layer.Gate)
                layer.Gate = dlarray(zeros(numChannels, 1));
            end
        end

        function Z = predict(layer, X)
            Z = sigmoid(layer.Gate) .* X;
        end
    end
end