function [] = plotPlanes(nname, saveyesno, PlanesTRJ,PlanesMetric)
    % Planes Metric

    points = extractdata(PlanesTRJ)';
    errors = extractdata(PlanesMetric)';
    figure;
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    

    % --- XZ Plane ---
    ax  = nexttile;
    y0  = 0;
    tol = 1e-6;

    % Filter points in XZ plane
    is_xz     = abs(points(:, 2) - y0) < tol;
    xz_points = points(is_xz, [1, 3]);
    xz_errors = errors(is_xz);

    % Grid
    x_vals = unique(xz_points(:, 1));
    z_vals = unique(xz_points(:, 2));
    [X, Zgrid] = meshgrid(x_vals, z_vals);
    Err = nan(size(X));

    % Assign errors
    [~, xi] = ismember(xz_points(:, 1), x_vals);
    [~, zi] = ismember(xz_points(:, 2), z_vals);
    for i = 1:length(xz_errors)
        Err(zi(i), xi(i)) = xz_errors(i);
    end

    % Step 1: Transform data
    logErr = log10(Err);
    logErr(Err <= 0) = NaN;

    % Step 2: Plot using surf with log10 data
    surf(X, Zgrid, zeros(size(logErr)), logErr, 'EdgeColor', 'none');
    view(2);
    axis equal tight;
    axis off;
    xlim ([-3.5 3.5])
    ylim ([-3.5 3.5])
    colormap jet;
    grid off;

    % Step 3: Set color axis and colorbar ticks
    clim([-2 2])

    text(0.5, -0.05, 0, 'XZ Plane'         , ...
        'Units'              , 'normalized', ...
        'HorizontalAlignment', 'center'    , ...
        'FontSize'           , 20          , ...
        'FontWeight'         , 'bold'      , ...
        'Parent'             , ax            ...
    )


    % --- XY Plane ---
    ax  = nexttile;
    z0  = 0;
    tol = 1e-6;

    % Filter points in XY plane
    is_xy     = abs(points(:, 3) - z0) < tol;
    xy_points = points(is_xy, 1:2);
    xy_errors = errors(is_xy);

    % Grid
    x_vals = unique(xy_points(:, 1));
    y_vals = unique(xy_points(:, 2));
    [X, Y] = meshgrid(x_vals, y_vals);
    Err    = nan(size(X));

    % Assign errors to Z
    [~, xi] = ismember(xy_points(:, 1), x_vals);
    [~, yi] = ismember(xy_points(:, 2), y_vals);
    for i = 1:length(xy_errors)
        Err(yi(i), xi(i)) = xy_errors(i);
    end

    % Step 1: Transform data
    logErr = log10(Err);
    logErr(Err <= 0) = NaN;

    % Step 2: Plot using surf with log10 data
    surf(X, Y, zeros(size(logErr)), logErr, 'EdgeColor', 'none');
    view(2);
    xlim ([-3.5 3.5])
    ylim ([-3.5 3.5])
    axis equal tight;
    axis off;
    colormap jet;
    grid off;

    % Step 3: Set color axis and colorbar ticks
    clim([-2 2])

    text(0.5, -0.05, 0, 'XY Plane'         , ...
        'Units'              , 'normalized', ...
        'HorizontalAlignment', 'center'    , ...
        'FontSize'           , 20          , ...
        'FontWeight'         , 'bold'      , ...
        'Parent'             , ax            ...
    )
   

    % --- YZ Plane ---
    ax  = nexttile;
    x0  = 0;
    tol = 1e-6;

    % Filter points in YZ plane
    is_yz     = abs(points(:, 1) - x0) < tol;
    yz_points = points(is_yz, [2, 3]);
    yz_errors = errors(is_yz);

    % Grid
    y_vals = unique(yz_points(:, 1));
    z_vals = unique(yz_points(:, 2));
    [Y, Zgrid] = meshgrid(y_vals, z_vals);
    Err = nan(size(Y));

    % Assign errors
    [~, yi] = ismember(yz_points(:, 1), y_vals);
    [~, zi] = ismember(yz_points(:, 2), z_vals);
    for i = 1:length(yz_errors)
        Err(zi(i), yi(i)) = yz_errors(i);
    end

    % Step 1: Transform data
    logErr = log10(Err);
    logErr(Err <= 0) = NaN;

    % Step 2: Plot using surf with log10 data
    surf(Y, Zgrid, zeros(size(logErr)), logErr, 'EdgeColor', 'none');
    view(2);
    axis equal tight;
    axis off;
    xlim ([-3.5 3.5])
    ylim ([-3.5 3.5])
    colormap jet;
    grid off;

    % Step 3: Set color axis and colorbar ticks
    clim([-2 2])

    text(0.5, -0.05, 0, 'YZ Plane'         , ...
        'Units'              , 'normalized', ...
        'HorizontalAlignment', 'center'    , ...
        'FontSize'           , 20          , ...
        'FontWeight'         , 'bold'      , ...
        'Parent'             , ax            ...
    )

    
    cb            = colorbar;
    log_ticks     = -2:1:2;
    cb.Ticks      = log_ticks;
    cb.TickLabels = arrayfun(@(x) sprintf('10^{%d}', x), log_ticks, 'UniformOutput', false);
    ylabel(cb, 'Percent Error', 'FontWeight', 'bold', 'FontSize', 12)
    cb.FontName   = 'Palatino Linotype';
    set(colorbar,'visible','off')

    
    if true == saveyesno
        exportgraphics(gcf, "../../fig/" + nname + "/PLN_" + nname + ".png", 'Resolution', 300);
    end
end