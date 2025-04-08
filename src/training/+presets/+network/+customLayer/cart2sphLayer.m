classdef cart2sphLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    methods
        function layer = cart2sphLayer(args)
            arguments
                args.Name        = "cart2sphLayer";
                args.Description = "Convert cartesian coordinates to spherical coordinates";
                args.InputNames  = "Trajectory";
                args.OutputNames = ["Spherical", "Radius"];
            end
            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function [Spherical, Radius] = predict(~, Trajectory)
            % Get cartesian coordinates
            [x, y, z] = deal(Trajectory(1, :), Trajectory(2, :), Trajectory(3, :));

            % Convert to spherical coordinates
            Radius    = sqrt(x .^ 2 + y .^ 2 + z .^ 2);
            [s, t, u] = deal(sin(x ./ Radius), sin(y ./ Radius), sin(z ./ Radius));
            [ri, re]  = deal(min(Radius, 1), 1 ./ max(Radius, 1));
            Spherical = cat(1, ri, re, s, t, u);
        end
    end
end