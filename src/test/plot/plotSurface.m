function [] = plotSurface(nname, saveyesno, SurfaceTRJ,SurfaceMetric)
    % Surface Metric

    figure;
    points    = extractdata(SurfaceTRJ)';
    errors    = extractdata(SurfaceMetric)';
    logerrors = log10(errors);
    
    ax   = subplot(1, 3, 1);
    lims = [-1 1];

    % 1
    scatter3(points(:, 1), points(:, 2), points(:, 3), 25, logerrors, "filled");
    clim([-2 2])
    view(2);
    axis equal tight padded;
    axis off;
    colormap jet;
    grid off;
    xlim(lims)
    ylim(lims)
    zlim(lims)

    text(0.5, 0.20, 0, 'Top View'          , ...
        'Units'              , 'normalized', ...
        'HorizontalAlignment', 'center'    , ...
        'FontSize'           , 14          , ...
        'FontWeight'         , 'bold'      , ...
        'Parent'             , ax            ...
    )


    % Expand limits
    xlim_current = xlim;
    ylim_current = ylim;
    zlim_current = zlim; 
    expand_ratio = 0.3;
    x_range      = diff(xlim_current);
    y_range      = diff(ylim_current);
    z_range      = diff(zlim_current);
    xlim([xlim_current(1) - expand_ratio * x_range, xlim_current(2) + expand_ratio * x_range])
    ylim([ylim_current(1) - expand_ratio * y_range, ylim_current(2) + expand_ratio * y_range])
    zlim([zlim_current(1) - expand_ratio * z_range, zlim_current(2) + expand_ratio * z_range]) 

    view(0, 180);


    % 2
    ax = subplot(1, 3, 2);
    scatter3(points(:, 1), points(:, 2), points(:, 3), 25, logerrors, "filled");
    clim([-2 2])
    view(2);
    axis equal padded;
    axis off;
    colormap jet;
    grid off;
    xlim(lims)
    ylim(lims)
    zlim(lims)
    
    ax.Position(1) = ax.Position(1) - 0.08;
    
    text(0.5, 0.20, 0, 'Front View'        , ...
        'Units'              , 'normalized', ...
        'HorizontalAlignment', 'center'    , ...
        'FontSize'           , 14          , ...
        'FontWeight'         , 'bold'      , ...
        'Parent'             , ax            ...
    )

    % Expand limits
    xlim_current = xlim;
    ylim_current = ylim;
    zlim_current = zlim; 
    expand_ratio = 0.3;
    x_range      = diff(xlim_current);
    y_range      = diff(ylim_current);
    z_range      = diff(zlim_current);
    xlim([xlim_current(1) - expand_ratio * x_range, xlim_current(2) + expand_ratio * x_range])
    ylim([ylim_current(1) - expand_ratio * y_range, ylim_current(2) + expand_ratio * y_range])
    zlim([zlim_current(1) - expand_ratio * z_range, zlim_current(2) + expand_ratio * z_range]) 
    view(0, 90);


    % 3
    ax = subplot(1, 3, 3);
    R             = [0 0 1; 0 1 0; -1 0 0]; % Rotate -90° around Y
    rotatedpoints = (R * points')';
    scatter3(rotatedpoints(:, 1), rotatedpoints(:, 2), rotatedpoints(:, 3), 25, logerrors, "filled");
    clim([-2 2])
    cb = colorbar;
    view(2);
    axis equal padded;
    axis off;
    colormap jet;
    grid off;
    xlim(lims)
    ylim(lims)
    zlim(lims)
    
    ax.Position(1) = ax.Position(1) - 0.15;

    text(0.5, 0.08, 0, 'Side View'         , ...
        'Units'              , 'normalized', ...
        'HorizontalAlignment', 'center'    , ...
        'FontSize'           , 14          , ...
        'FontWeight'         , 'bold'      , ...
        'Parent'             , ax            ...
    )

    % Expand limits
    xlim_current = xlim;
    ylim_current = ylim;
    zlim_current = zlim; 
    expand_ratio = 0.1;
    x_range      = diff(xlim_current);
    y_range      = diff(ylim_current);
    z_range      = diff(zlim_current);
    xlim([xlim_current(1) - expand_ratio * x_range, xlim_current(2) + expand_ratio * x_range])
    ylim([ylim_current(1) - expand_ratio * y_range, ylim_current(2) + expand_ratio * y_range])
    zlim([zlim_current(1) - expand_ratio * z_range, zlim_current(2) + expand_ratio * z_range]) 

    
    log_ticks   = -2:1:2;
    cb.Ticks    = log_ticks;
    ylabel(cb, 'Percent Error', 'FontWeight', 'bold', 'FontSize', 12)
    cb.FontName = 'Palatino Linotype';
    set(colorbar,'visible','off')
    view(0, 90);


    if true == saveyesno
        exportgraphics(gcf, "../../fig/" + nname + "/SRF_" + nname + ".png", 'Resolution', 300);
    end
end