classdef cart2SphLayer < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % cart2SphLayer Convert Cartesian coordinates to Spherical coordinates
    % Uses Pines formulation for non-singular spherical coordinates representation.
    % Converts [x,y,z] to [ri,re,s,t,u].
    %
    % cart2SphLayer Methods:
    %    predict - Computes spherical [ri,re,s,t,u] coordinates, and presents secondary output for layers that require direct Radius access
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
            r   = vecnorm(Trajectory);
            stu = Trajectory ./ r ;
            stu(~isfinite(stu)) = 0;
            ri  = min(r, 1);
            re  = 1 ./ max(r, 1);

            Spherical = cat(1, ri, re, stu);
            Radius    = r;
        end
    end
end
