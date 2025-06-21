classdef cart2sphLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % cart2sphLayer Convert Cartesian coordinates to Spherical coordinates
    % Uses Pines formulation for non-singular spherical coordinates representation.
    % Converts [x,y,z] to [ri,re,s,t,u].
    %
    % cart2sphLayer Methods:
    %    predict - Computes spherical [ri,re,s,t,u] coordinates, and presents secondary output for layers that require direct Radius access
    %
    % See also presets.layer.analyticModelLayer, presets.layer.applyBoundaryConditionsLayer.
    
    methods
        function layer = cart2sphLayer(args)
            arguments
                args.Name        = "cart2sphLayer"
                args.InputNames  = "Trajectory"
                args.OutputNames = ["Spherical", "Radius"]
            end

            layer.Name        = args.Name;
            layer.InputNames  = args.InputNames;
            layer.OutputNames = args.OutputNames;
        end

        function [Spherical, Radius] = predict(~, Trajectory)
            [x, y, z] = deal(Trajectory(1, :), Trajectory(2, :), Trajectory(3, :));
            
            Radius    = sqrt(x .^ 2 + y .^ 2 + z .^ 2);
            [s, t, u] = deal(sin(x ./ Radius), sin(y ./ Radius), sin(z ./ Radius));
            [ri, re]  = deal(min(Radius, 1), 1 ./ max(Radius, 1));
            Spherical = cat(1, ri, re, s, t, u);
        end
    end
end