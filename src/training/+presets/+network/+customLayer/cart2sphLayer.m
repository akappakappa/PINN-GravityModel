classdef cart2sphLayer < nnet.layer.Layer
    properties
        RotationMatrix
        ScaleFactor
    end
    methods
        function layer = cart2sphLayer()
            layer.Name = "cart2sphLayer";
            layer.RotationMatrix = [];
            layer.ScaleFactor = 1;
        end
        function SPH = predict(layer, TRJ)
            x = TRJ(1);
            y = TRJ(2);
            z = TRJ(3);
            % Trajectory (cartesian to spherical)
            [theta, phi, r] = cart2sph(x, y, z);
            s = sin(x ./ r);
            t = sin(y ./ r);
            u = sin(z ./ r);

            ri          = r;
            ri(ri >= 1) = 1;
            re          = r;
            re(re <= 1) = 1;
            re(re > 1)  = 1 ./ re(re > 1);

            SPH = {ri, re, s, t, u};

            % Acceleration (rotate)
            s_theta = sin(theta);
            c_theta = cos(theta);
            s_phi   = sin(phi);
            c_phi   = cos(phi);

            r_hat     = [s_phi .* c_theta, s_phi .* s_theta, c_phi];
            theta_hat = [c_phi .* c_theta, c_phi .* s_theta, -s_phi];
            phi_hat   = [-s_theta, c_theta, zeros(size(s_theta))];

            layer.RotationMatrix = cat(3, r_hat, theta_hat, phi_hat);
            layer.RotationMatrix = permute(layer.RotationMatrix, [2, 3, 1]);

            % Potential (proxy)
            layer.ScaleFactor = r;
            layer.ScaleFactor(layer.ScaleFactor <= 1) = 1;
        end
    end
end