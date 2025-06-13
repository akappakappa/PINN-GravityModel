classdef c2sRadiusLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % c2sRadiusLayer Convert cartesian coordinates to spherical coordinates.
    %   This layer computes the spherical coordinates as [ri, re, s, t, u] vector.

    methods
        function layer = c2sRadiusLayer(args)
            arguments
                args.Name        = "c2sRadiusLayer";
                args.Description = "Convert cartesian coordinates to spherical coordinates";
                args.InputNames  = "Trajectory";
                args.OutputNames = "Spherical";
            end
            % Construct the layer.

            layer.Name        = args.Name;
            layer.Description = args.Description;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function Spherical = predict(~, Trajectory)
            % Computes the [r, s, t, u] spherical coordinates at the given [x, y, z] TRAJECTORY points.

            % Get cartesian coordinates
            [x, y, z] = deal(Trajectory(1, :), Trajectory(2, :), Trajectory(3, :));

            % Convert to spherical coordinates
            r         = sqrt(x .^ 2 + y .^ 2 + z .^ 2);
            [s, t, u] = deal(sin(x ./ r), sin(y ./ r), sin(z ./ r));
            Spherical = cat(1, r, s, t, u);
        end
    end
end