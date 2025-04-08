function [Spherical, Radius] = cart2sphLayer(Trajectory)
    % Get cartesian coordinates
    [x, y, z] = deal(Trajectory(1, :), Trajectory(2, :), Trajectory(3, :));

    % Convert to spherical coordinates
    Radius    = vecnorm(Trajectory, 2, 1);
    [s, t, u] = deal(sin(x ./ Radius), sin(y ./ Radius), sin(z ./ Radius));
    [ri, re]  = deal(min(Radius, 1), 1 ./ max(Radius, 1));
    Spherical = [ri; re; s; t; u];
end