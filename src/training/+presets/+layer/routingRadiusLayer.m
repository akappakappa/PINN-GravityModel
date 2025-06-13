classdef routingRadiusLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    properties (Learnable)
        threshold
        width
    end

    methods
        function layer = routingRadiusLayer(args)
            arguments
                args.Name        = "routingRadiusLayer";
                args.Description = "Routes the input through a different layer based on the radius value and probability distribution";
                args.InputNames  = "Spherical";
                args.OutputNames = ["RouteHigher", "RouteLower", "RadiusHigher", "RadiusLower", "RouteIndexes", "Radius"];
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
            layer.threshold   = 1;
            layer.width       = 0.5;
        end

        function [RouteHigher, RouteLower, RadiusHigher, RadiusLower, RouteIndexes, Radius] = predict(layer, Spherical)
            % Computes the routing based on the spherical coordinates.
            % The routing is based on the radius value and a probability distribution.

            % Extract radius from spherical coordinates
            Radius = Spherical(1, :);

            % Probabilistic routing
            probThreshold = 1 ./ (1 + exp(-(Radius - layer.threshold) / (layer.width / 10)));
            randomNumbers = rand(size(Radius));
            RouteIndexes  = randomNumbers < probThreshold;
            RouteHigher   = Spherical(:, RouteIndexes);
            RadiusHigher  = Radius(:, RouteIndexes);
            RouteLower    = Spherical(:, ~RouteIndexes);
            RadiusLower   = Radius(:, ~RouteIndexes);
        end
    end
end