function [SPH, Radius] = cart2sphLayer(TRJ)
    % Trajectory (cartesian to spherical)
    x = TRJ(1, :);
    y = TRJ(2, :);
    z = TRJ(3, :);

    % Replacing to keep tracing for dlgradient: [theta, phi, r] = cart2sph(x, y, z);
    r = vecnorm(TRJ, 2, 1);

    s = sin(x ./ r);
    t = sin(y ./ r);
    u = sin(z ./ r);
    
    ri = min(r, 1);
    re = max(r, 1);
    re = 1 ./ re;

    SPH = [ri; re; s; t; u];

    % Potential (proxy)
    Radius = r;
end