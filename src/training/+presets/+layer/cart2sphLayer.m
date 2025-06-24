classdef cart2SphLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % cart2SphLayer Convert Cartesian coordinates to Spherical coordinates
    % Uses Pines formulation for non-singular spherical coordinates representation.
    % Converts [x,y,z] to [ri,re,s,t,u].
    %
    % cart2SphLayer Methods:
    %    predict - Computes spherical [ri,re,s,t,u] coordinates, and presents secondary and tertiary output for layers that require direct Radius access
    %
    % See also presets.layer.analyticModelLayer, presets.layer.applyBoundaryConditionsLayer.
    
    methods
        function layer = cart2SphLayer(args)
            arguments
                args.Name        = "cart2SphLayer"
                args.InputNames  = "Trajectory"
                args.OutputNames = ["Spherical", "Radius"]
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function [Spherical, Radius] = predict(~, Trajectory)
            [x, y, z] = deal(Trajectory(1, :), Trajectory(2, :), Trajectory(3, :));
            
            r         = sqrt(x .^ 2 + y .^ 2 + z .^ 2);
            [s, t, u] = deal(x ./ r, y ./ r, z ./ r);
            [ri, re]  = deal(min(r, 1), 1 ./ max(r, 1));

            Spherical = cat(1, ri, re, s, t, u);
            Radius    = r;
        end
    end
end