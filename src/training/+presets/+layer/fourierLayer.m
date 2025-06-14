classdef fourierLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % fourierLayer Fourier Features Layer
    %   This layer implements Fourier Features.

    properties (Learnable)
        B   % Projection matrix
    end

    properties
        InputSize
        OutputSize
        Scale   % Frequency scale
    end

    methods
        function layer = fourierLayer(InputSize, OutputSize, Scale, args)
            arguments
                InputSize
                OutputSize
                Scale
                args.Name        = "fourierLayer";
                args.Description = "Fourier Features Layer";
            end

            layer.Name        = args.Name;
            layer.InputSize   = InputSize;
            layer.OutputSize  = OutputSize;
            layer.Description = args.Description;
            layer.Scale       = Scale;

            layer.B = dlarray(Scale .* randn(OutputSize / 2, InputSize));
        end

        function Z = predict(layer, X)
            BX = dlarray(layer.B * stripdims(X), "CB");
            Z  = cat(1, sin(BX), cos(BX));
        end
    end

end