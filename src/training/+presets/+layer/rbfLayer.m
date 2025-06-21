classdef rbfLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % rbfLayer Radial Basis Function (RBF)
    % Apply Radial Basis Function (RBF) to the input with learnable centers.
    %
    % rbfLayer Properties:
    %    Centers    - Center points
    %    Gamma      - Scaling parameters vector
    %    OutputSize - Number of output neurons
    %
    % rbfLayer Methods:
    %    initialize - Random Centers initialization, Ones Gamma initialization
    %    predict    - Computes RBF betweem input X and Centers, scaling with Gamma

    properties (Learnable)
        Centers
        Gamma
    end

    properties
        OutputSize
    end
    
    methods
        function layer = rbfLayer(OutputSize, args)
            arguments
                OutputSize
                args.Name    = "rbfLayer"
                args.Centers = []
                args.Gamma   = []
            end
            
            layer.Name = args.Name;

            layer.OutputSize = OutputSize;
            layer.Centers    = args.Centers;
            layer.Gamma      = args.Gamma;
        end
        function layer = initialize(layer, layout)
            inChannels  = layout.Size(finddim(layout, "C"));
            outChannels = layer.OutputSize;
            
            if isempty(layer.Centers)
                layer.Centers = dlarray(rand(inChannels, outChannels));
            end
            
            if isempty(layer.Gamma)
                layer.Gamma   = dlarray(ones(outChannels, 1));
            end
        end
        
        function Z = predict(layer, X)
            C2    = sum(layer.Centers .^ 2, 1);
            X2    = sum(stripdims(X) .^ 2, 1);
            XC    = stripdims(X)' * layer.Centers;
            Dists = dlarray(((X2' + C2) - 2 * XC)', "CB");

            Z = exp(-layer.Gamma .* Dists);
        end
    end
end