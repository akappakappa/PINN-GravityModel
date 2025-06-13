classdef mergingRadiusLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable

    methods
        function layer = mergingRadiusLayer(args)
            arguments
                args.Name        = "mergingRadiusLayer";
                args.Description = "Merges routes based on indexes";
                args.InputNames  = ["PotHigher", "PotLower", "RouteIndexes"];
                args.OutputNames = "Potential";
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Potential = predict(~, PotHigher, PotLower, RouteIndexes)
            % Merges the higher and lower routes based on the provided indexes.
            % The output is the potential at the given route indexes.

            % Matlab can't initialize properly (bug)
            if isempty(PotHigher) && ~isempty(PotLower)
                if PotLower == dlarray([1], "CB")
                    temp      = PotHigher;
                    PotHigher = PotLower;
                    PotLower  = temp;
                end
            end

            Potential                = dlarray(zeros(size(RouteIndexes)), "CB");
            Potential(RouteIndexes)  = PotHigher;
            Potential(~RouteIndexes) = PotLower;
        end
    end
end