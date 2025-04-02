classdef cart2sphLayer < nnet.layer.Layer
    properties (State)
        Radius
    end
    methods
        function layer = cart2sphLayer(Name)
            arguments
                Name {mustBeText} = "cart2sphLayer"
            end
            layer.Name = Name;
        end
        function [SPH, Radius] = predict(layer, TRJ)
            % Trajectory (cartesian to spherical)
            x = TRJ(1, :);
            y = TRJ(2, :);
            z = TRJ(3, :);

            % Replacing to keep tracing for dlgradient: [theta, phi, r] = cart2sph(x, y, z);
            r = vecnorm(TRJ, 2, 1);

            s = sin(x ./ r);
            t = sin(y ./ r);
            u = sin(z ./ r);
            
            ri          = r;
            ri(ri >= 1) = 1;
            re          = r;
            re(re <= 1) = 1;
            re(re > 1)  = 1 ./ re(re > 1);

            SPH = [ri; re; s; t; u];

            % Potential (proxy)
            Radius = r;
            layer.Radius = r;
        end
    end
end