function [] = plotPlanes(PlanesTRJ,PlanesMetric)
    % Planes Metric
    points = extractdata(PlanesTRJ)';
    errors = extractdata(PlanesMetric)';
    figure;  % [left bottom width height]
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % XY Plane
    nexttile

    % Parameters
    z0 = 0;            % Choose the z-slice
    tol = 1e-6;        % Tolerance to filter points on the plane

    % Filter points in XY plane
    is_xy = abs(points(:,3) - z0) < tol;
    xy_points = points(is_xy, 1:2);         % Keep x and y
    xy_errors = errors(is_xy);

    % Grid
    x_vals = unique(xy_points(:,1));
    y_vals = unique(xy_points(:,2));
    [X, Y] = meshgrid(x_vals, y_vals);
    Err = nan(size(X));

    % Assign errors to Z
    [~, xi] = ismember(xy_points(:,1), x_vals);
    [~, yi] = ismember(xy_points(:,2), y_vals);
    for i = 1:length(xy_errors)
        Err(yi(i), xi(i)) = xy_errors(i);
    end

    % Step 1: Transform data
    logErr = log10(Err);
    logErr(Err <= 0) = NaN;  % Avoid log10 of zero or negative numbers

    % Step 2: Plot using surf with log10 data
    surf(X, Y, zeros(size(logErr)), logErr, 'EdgeColor', 'none');
    view(2);
    axis equal tight;
    colormap jet;
    grid off;

    % Step 3: Set color axis and colorbar ticks
    clim([-2 2])  % log10 scale from 10^-3 to 10^3
   
    % YZ Plane
    % Parameters
    %figure;
    %subplot(1,3,2);
    nexttile

    x0 = 0;
    tol = 1e-6;

    % Filter points in YZ plane
    is_yz = abs(points(:,1) - x0) < tol;
    yz_points = points(is_yz, [2,3]);       % Keep y and z
    yz_errors = errors(is_yz);

    % Grid
    y_vals = unique(yz_points(:,1));
    z_vals = unique(yz_points(:,2));
    [Y, Zgrid] = meshgrid(y_vals, z_vals);
    Err = nan(size(Y));

    % Assign errors
    [~, yi] = ismember(yz_points(:,1), y_vals);
    [~, zi] = ismember(yz_points(:,2), z_vals);
    for i = 1:length(yz_errors)
        Err(zi(i), yi(i)) = yz_errors(i);
    end

    % Step 1: Transform data
    logErr = log10(Err);
    logErr(Err <= 0) = NaN;  % Avoid log10 of zero or negative numbers

    % Step 2: Plot using surf with log10 data
    surf(Y, Zgrid, zeros(size(logErr)), logErr, 'EdgeColor', 'none');
    view(2);
    axis equal tight;
    colormap jet;
    grid off;

    % Step 3: Set color axis and colorbar ticks
    clim([-2 2])  % log10 scale from 10^-3 to 10^3
    

    % XZ Plane
    %figure;
    %subplot(1,3,3);
    nexttile

    % Parameters
    y0 = 0;
    tol = 1e-6;

    % Filter points in XZ plane
    is_xz = abs(points(:,2) - y0) < tol;
    xz_points = points(is_xz, [1,3]);       % Keep x and z
    xz_errors = errors(is_xz);

    % Grid
    x_vals = unique(xz_points(:,1));
    z_vals = unique(xz_points(:,2));
    [X, Zgrid] = meshgrid(x_vals, z_vals);
    Err = nan(size(X));

    % Assign errors
    [~, xi] = ismember(xz_points(:,1), x_vals);
    [~, zi] = ismember(xz_points(:,2), z_vals);
    for i = 1:length(xz_errors)
        Err(zi(i), xi(i)) = xz_errors(i);
    end

    % Step 1: Transform data
    logErr = log10(Err);
    logErr(Err <= 0) = NaN;  % Avoid log10 of zero or negative numbers

    % Step 2: Plot using surf with log10 data
    surf(X, Zgrid, zeros(size(logErr)), logErr, 'EdgeColor', 'none');
    view(2);
    axis equal tight;
    colormap jet;
    grid off;
    % Step 3: Set color axis and colorbar ticks
    clim([-2 2])  % log10 scale from 10^-3 to 10^3

    cb = colorbar;
    log_ticks = -2:1:2;
    cb.Ticks = log_ticks;
    cb.TickLabels = arrayfun(@(x) sprintf('10^{%d}', x), log_ticks, 'UniformOutput', false);
    ylabel(cb, 'Error Magnitude (log scale)', 'FontWeight', 'bold')

end