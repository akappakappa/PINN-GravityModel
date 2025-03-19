classdef testLayer < nnet.layer.Layer
    properties (State)
        RotationMatrix
        ScaleFactor
    end
    methods
        function layer = testLayer(Name)
            arguments
                Name {mustBeText} = "cart2sphLayer"
            end
            layer.Name = Name;
        end
        function [SPH, RotationMatrix, ScaleFactor] = predict(layer, TRJ)
            % Trajectory (cartesian to spherical)
            x = TRJ(1, :);
            y = TRJ(2, :);
            z = TRJ(3, :);

            r = vecnorm(TRJ, 2, 1);
            theta = atan2(y, x);
            phi = atan2(sqrt((x .^ 2) + (y .^ 2)), z);

            %[theta, phi, r] = cart2sph(x, y, z);
            s = sin(x ./ r);
            t = sin(y ./ r);
            u = sin(z ./ r);

            ri          = r;
            ri(ri >= 1) = 1;
            re          = r;
            re(re <= 1) = 1;
            re(re > 1)  = 1 ./ re(re > 1);

            SPH = [ri; re; s; t; u];

            % Acceleration (rotate)
            s_theta = sin(theta);
            c_theta = cos(theta);
            s_phi   = sin(phi);
            c_phi   = cos(phi);

            r_hat     = [s_phi .* c_theta; s_phi .* s_theta; c_phi];
            theta_hat = [c_phi .* c_theta; c_phi .* s_theta; -s_phi];
            phi_hat   = [-s_theta; c_theta; zeros(size(s_theta))];

            RotationMatrix = cat(3, r_hat, theta_hat, phi_hat);
            RotationMatrix = permute(RotationMatrix, [1, 3, 2]);
            layer.RotationMatrix = RotationMatrix;
            
            % Potential (proxy)
            ScaleFactor = r;
            ScaleFactor(ScaleFactor <= 1) = 1;
            layer.ScaleFactor = ScaleFactor;
        end
    end
end