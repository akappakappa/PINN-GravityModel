function [] = plotSurface(SurfaceTRJ,SurfaceMetric)
    % Surface Metric
    points = extractdata(SurfaceTRJ)';
    errors = extractdata(SurfaceMetric)';
    logerrors = log10(errors);
    figure;
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    lims = [-1 1];
    % Front view
    nexttile;
    scatter3(points(:,1), points(:,2), points(:,3), 25, logerrors, "filled");
    clim([-2 2])  % log10 scale from 10^-3 to 10^3
    view(2);
    axis equal tight padded;
    colormap jet;
    grid off;
    xlim(lims)
    ylim(lims)
    zlim(lims)

    % Get current limits
    xlim_current = xlim;
    ylim_current = ylim;
    zlim_current = zlim; 

    % Expand each limit by a percentage (e.g. 10%)
    expand_ratio = 0.3;
    x_range = diff(xlim_current);
    y_range = diff(ylim_current);
    z_range = diff(zlim_current);
    xlim([xlim_current(1) - expand_ratio*x_range, xlim_current(2) + expand_ratio*x_range])
    ylim([ylim_current(1) - expand_ratio*y_range, ylim_current(2) + expand_ratio*y_range])
    zlim([zlim_current(1) - expand_ratio*z_range, zlim_current(2) + expand_ratio*z_range]) 

    view(0,180);

    % Side view
    nexttile;
    scatter3(points(:,1), points(:,2), points(:,3), 25, logerrors, "filled");
    clim([-2 2])  % log10 scale from 10^-3 to 10^3
    view(2);
    axis equal padded;
    colormap jet;
    grid off;
    xlim(lims)
    ylim(lims)
    zlim(lims)

    % Get current limits
    xlim_current = xlim;
    ylim_current = ylim;
    zlim_current = zlim; 

    % Expand each limit by a percentage (e.g. 10%)
    expand_ratio = 0.3;
    x_range = diff(xlim_current);
    y_range = diff(ylim_current);
    z_range = diff(zlim_current);
    xlim([xlim_current(1) - expand_ratio*x_range, xlim_current(2) + expand_ratio*x_range])
    ylim([ylim_current(1) - expand_ratio*y_range, ylim_current(2) + expand_ratio*y_range])
    zlim([zlim_current(1) - expand_ratio*z_range, zlim_current(2) + expand_ratio*z_range]) 
    view(0,90);

    % Top view
    nexttile;
    R = [0 0 1; 0 1 0; -1 0 0];  % Rotate -90° around Y
    rotatedpoints = (R * points')';
    scatter3(rotatedpoints(:,1), rotatedpoints(:,2), rotatedpoints(:,3), 25, logerrors, "filled");
    clim([-2 2])  % log10 scale from 10^-3 to 10^3
    cb = colorbar;
    view(2);
    axis equal padded;
    colormap jet;
    grid off;
    xlim(lims)
    ylim(lims)
    zlim(lims)

    % Get current limits
    xlim_current = xlim;
    ylim_current = ylim;
    zlim_current = zlim; 

    % Expand each limit by a percentage (e.g. 10%)
    expand_ratio = 0.1;
    x_range = diff(xlim_current);
    y_range = diff(ylim_current);
    z_range = diff(zlim_current);
    xlim([xlim_current(1) - expand_ratio*x_range, xlim_current(2) + expand_ratio*x_range])
    ylim([ylim_current(1) - expand_ratio*y_range, ylim_current(2) + expand_ratio*y_range])
    zlim([zlim_current(1) - expand_ratio*z_range, zlim_current(2) + expand_ratio*z_range]) 

    log_ticks = -2:1:2;
    cb.Ticks = log_ticks;
    cb.TickLabels = arrayfun(@(x) sprintf('10^{%d}', x), log_ticks, 'UniformOutput', false);
    ylabel(cb, 'Error Magnitude (log scale)', 'FontWeight', 'bold')
    view(0,90);
end