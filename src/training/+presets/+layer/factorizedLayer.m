classdef factorizedLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % factorizedLayer Low-rank factorization approximation of a fully-connected layer.
    % Low-rank factorization of the weight matrix into two smaller matrices W1 and W2, such that W ~ W1 * W2.
    % Reduces number of learnable parameters, while trying to keep comparable level of expression.
    %
    % factorizedLayer Properties:
    %    W1, W2             - Learnable low-rank factorized weight matrices
    %    Bias               - Learnable bias vector
    %    OutputSize         - Number of output neurons
    %    Rank               - Factorization rank
    %    WeightsInitializer - Weights initialization as "glorot" (default) or "zeros"
    %
    % factorizedLayer Methods:
    %    initialize - "glorot" (default) or "zeros" weights initialization and zeros bias initialization
    %    predict    - Fully connect the layer by forming a new matrix W = W1 * W2
    
    properties (Learnable)
        W1
        W2
        Bias
    end
    
    properties
        OutputSize
        Rank
        WeightsInitializer
    end
    
    methods
        function layer = factorizedLayer(OutputSize, Rank, args)
            arguments
                OutputSize
                Rank

                args.Name = "factorizedLayer"
                
                args.W1   = []
                args.W2   = []
                args.Bias = []
                args.WeightsInitializer {mustBeMember(args.WeightsInitializer, {'glorot', 'zeros'})} = "glorot";
            end

            if xor(~isempty(args.W1), ~isempty(args.W2))
                error("factorizedLayer:InvalidInput", "W1 and W2 must be specified together (both non empty) or omitted.");
            end

            layer.Name = args.Name;

            layer.OutputSize         = OutputSize;
            layer.Rank               = Rank;
            layer.W1                 = args.W1;
            layer.W2                 = args.W2;
            layer.Bias               = args.Bias;
            layer.WeightsInitializer = args.WeightsInitializer;
        end

        function layer = initialize(layer, layout)
            inChannels  = layout.Size(finddim(layout, "C"));
            outChannels = layer.OutputSize;
            R           = layer.Rank;

            if (isempty(layer.W1) & isempty(layer.W2))
                switch layer.WeightsInitializer
                    case "glorot"
                        W = (sqrt(6 / (inChannels + outChannels))) .* ...
                            ( 2 * rand([outChannels, inChannels]) - 1 );
                        [U, S, V] = svds(W, R);
                        layer.W1  = dlarray(U * sqrt(S));
                        layer.W2  = dlarray(sqrt(S) * V');
                    case "zeros"
                        layer.W1  = dlarray(1e-3 * ones(outChannels, R));
                        layer.W2  = dlarray(1e-3 * ones(R, inChannels));
                end
            end

            if isempty(layer.Bias)
                layer.Bias = dlarray(zeros(outChannels, 1));
            end
        end
        
        function Z = predict(layer, X)
            Z = fullyconnect(X, layer.W1 * layer.W2, layer.Bias);
        end
    end
end