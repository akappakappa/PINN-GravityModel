% This script tests the performance of the trained model.
%
% File: runTest.m
%     entrypoint for Testing

metricsFolder   = "src/preprocessing/datastore/metrics/";
headless        = batchStartupOptionUsed;

% Preparations - Data
data = mLoadData("src/preprocessing/metricsData.mat");
net  = load("src/training/residual.mat").net;

% Compute metrics
% ---------------------------------------------------------- | Preset func -- | NN | Trajectory Data ------------- | Acceleration Data ----------- | Potential Data -------------- |
[PlanesMetric               , PlanesRadius        ] = dlfeval(@presets.mpeLoss, net, data.mPlanesTRJ               , data.mPlanesACC               , data.mPlanesPOT               );
[GeneralizationMetric_0_1   , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_0_1   , data.mGeneralizationACC_0_1   , data.mGeneralizationPOT_0_1   );
[GeneralizationMetric_1_10  , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_1_10  , data.mGeneralizationACC_1_10  , data.mGeneralizationPOT_1_10  );
[GeneralizationMetric_10_100, GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ_10_100, data.mGeneralizationACC_10_100, data.mGeneralizationPOT_10_100);
[GeneralizationMetric       , GeneralizationRadius] = dlfeval(@presets.mpeLoss, net, data.mGeneralizationTRJ       , data.mGeneralizationACC       , data.mGeneralizationPOT       );
[SurfaceMetric              , SurfaceRadius       ] = dlfeval(@presets.mpeLoss, net, data.mSurfaceTRJ              , data.mSurfaceACC              , data.mSurfacePOT              );

fprintf("\n### Mean Percent Error (MPE) ###\n");
fprintf("Planes metric                   : %f\n", mean(PlanesMetric               ));
fprintf("Generalization metric [0R:1R]   : %f\n", mean(GeneralizationMetric_0_1   ));
fprintf("Generalization metric [1R:10R]  : %f\n", mean(GeneralizationMetric_1_10  ));
fprintf("Generalization metric [10R:100R]: %f\n", mean(GeneralizationMetric_10_100));
fprintf("Generalization metric [0R:100R] : %f\n", mean(GeneralizationMetric       ));
fprintf("Surface metric                  : %f\n", mean(SurfaceMetric              ));

% Plotting
if headless
    return;
end

% Generalization: mpeLoss vs. distance(R), convert mpeLoss in log scale
figure;
semilogy(extractdata(GeneralizationRadius), extractdata(GeneralizationMetric), '.', 'DisplayName', 'Generalization');

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xlabel('Distance (R)');
ylabel('Mean Percent Error (MPE)');
title('Generalization: MPE vs. Distance (R)');
legend('show'); 

% Predicting potentials within the network
actNN               = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'scaleNNPotentialLayer');
actLF               = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'analyticModelLayer'   );
actFuse             = minibatchpredict(net, data.mGeneralizationTRJ, "Outputs", 'fuseModelsLayer'      );
[sortedRadius, idx] = sort(extractdata(GeneralizationRadius));
sortedNN            = abs(extractdata(actNN(idx)  ));
sortedLF            = abs(extractdata(actLF(idx)  ));
sortedFuse          = abs(extractdata(actFuse(idx)));

% Plotting potentials
figure;
hold on;
semilogy(sortedRadius, sortedLF  , '.', 'DisplayName', 'PotAnalytic');
semilogy(sortedRadius, sortedFuse, '.', 'DisplayName', 'PotFused'   );
semilogy(sortedRadius, sortedNN  , '.', 'DisplayName', 'PotNN'      );

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xline(10, '--', 'R = 10', 'LabelVerticalAlignment', 'bottom');
xlabel('Distance (R)');
ylabel('Potential');
title('Generalization Potential: Analytic vs Fused');
legend('show');

% NN vs Analytic (Fusion)
figure;
semilogy(sortedRadius, abs(sortedNN - sortedLF), '.', 'DisplayName', 'NN - Analytic');

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xline(10, '--', 'R = 10', 'LabelVerticalAlignment', 'bottom');
xlabel('Distance (R)');
ylabel('Absolute Difference');
title('Generalization Potential: Difference between NN and Analytic Potential');
legend('show');

% Fused vs Analytic (Boundary)
figure;
semilogy(sortedRadius, sortedNN, '.', 'DisplayName', 'NN (= Fused - Analytic)');

set(gca, 'YScale', 'log');
xlim([0, 20]);
grid on;
xline(10, '--', 'R = 10', 'LabelVerticalAlignment', 'bottom');
xlabel('Distance (R)');
ylabel('Absolute Difference');
title('Generalization Potential: NN');
legend('show');

% Planes Metric
points = extractdata(data.mPlanesTRJ)';
errors = extractdata(PlanesMetric)';

% XY Plane
% Parameters
figure;
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
view(2), axis equal tight;
colormap jet;
grid off;

% Step 3: Set color axis and colorbar ticks
clim([-2 2])  % log10 scale from 10^-3 to 10^3
cb = colorbar;
log_ticks = -2:1:2;
cb.Ticks = log_ticks;
cb.TickLabels = arrayfun(@(x) sprintf('10^{%d}', x), log_ticks, 'UniformOutput', false);
ylabel(cb, 'Error Magnitude (log scale)', 'FontWeight', 'bold')

% YZ Plane
% Parameters
figure;
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
view(2), axis equal tight;
colormap jet;
grid off;

% Step 3: Set color axis and colorbar ticks
clim([-2 2])  % log10 scale from 10^-3 to 10^3
cb = colorbar;
log_ticks = -2:1:2;
cb.Ticks = log_ticks;
cb.TickLabels = arrayfun(@(x) sprintf('10^{%d}', x), log_ticks, 'UniformOutput', false);
ylabel(cb, 'Error Magnitude (log scale)', 'FontWeight', 'bold')

% XZ Plane
figure;

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
view(2), axis equal tight;
colormap jet;
grid off;
% Step 3: Set color axis and colorbar ticks
clim([-2 2])  % log10 scale from 10^-3 to 10^3

cb = colorbar;
log_ticks = -2:1:2;
cb.Ticks = log_ticks;
cb.TickLabels = arrayfun(@(x) sprintf('10^{%d}', x), log_ticks, 'UniformOutput', false);
ylabel(cb, 'Error Magnitude (log scale)', 'FontWeight', 'bold')

% Surface Metric
points = extractdata(data.mSurfaceTRJ)';
errors = extractdata(SurfaceMetric)';
logerrors = log10(errors);

% Front view
figure;
scatter3(points(:,1), points(:,2), points(:,3), 25, logerrors, "filled");
clim([-2 2])  % log10 scale from 10^-3 to 10^3
cb = colorbar;
view(2), axis equal padded;
colormap jet;
grid off;

% Get current limits
xlim_current = xlim;
ylim_current = ylim;
zlim_current = zlim; 

% Expand each limit by a percentage (e.g. 10%)
expand_ratio = 0.2;
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
view(0,0);

% Side view
figure;
scatter3(points(:,1), points(:,2), points(:,3), 25, logerrors, "filled");
clim([-2 2])  % log10 scale from 10^-3 to 10^3
cb = colorbar;
view(2), axis equal padded;
colormap jet;
grid off;

% Get current limits
xlim_current = xlim;
ylim_current = ylim;
zlim_current = zlim; 

% Expand each limit by a percentage (e.g. 10%)
expand_ratio = 0.2;
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

% Top view
figure;
scatter3(points(:,1), points(:,2), points(:,3), 25, logerrors, "filled");
clim([-2 2])  % log10 scale from 10^-3 to 10^3
cb = colorbar;
view(2), axis equal padded;
colormap jet;
grid off;

% Get current limits
xlim_current = xlim;
ylim_current = ylim;
zlim_current = zlim; 

% Expand each limit by a percentage (e.g. 10%)
expand_ratio = 0.2;
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
view(90,0);
return;

% Planes: heatmap of 3 planes, value=mpeLoss(i), 2Dposition=PlanesTrj, planeID= which value of PlanesTrj(1:3,i) is 0
figure;
title('Planes: MPE');
XYi = 0 == PlanesTrj(3, :);
XZi = 0 == PlanesTrj(2, :);
YZi = 0 == PlanesTrj(1, :);
XY  = extractdata(PlanesTrj(1:2  , XYi));
XZ  = extractdata(PlanesTrj(1:2:3, XZi));
YZ  = extractdata(PlanesTrj(2:3  , YZi));
XYm = PlanesMetric(XYi);
XZm = PlanesMetric(XZi);
YZm = PlanesMetric(YZi);
subplot(1, 3, 1);
scatter(XY(1, :), XY(2, :), 2e2, XYm, '.');
title('XY Plane');
subplot(1, 3, 2);
scatter(XZ(1, :), XZ(2, :), 2e2, XZm, '.');
title('XZ Plane');
subplot(1, 3, 3);
scatter(YZ(1, :), YZ(2, :), 2e2, YZm, '.');
title('YZ Plane');
colormap jet;
colorbar;

% Surface: 3d plot of surface, with color value depending on the mpeLoss
figure;
scatter3(extractdata(SurfaceTrj(1, :)), extractdata(SurfaceTrj(2, :)), extractdata(SurfaceTrj(3, :)), 2e2, SurfaceMetric, 'o', 'filled');
title('Surface: MPE');

clearvars -except DO_DATA_EXTRACTION DO_PREPROCESSING DO_TRAINING DO_TESTING



function data = mLoadData(path)
    data = load(path);
    data.mGeneralizationTRJ = cat(1, data.mGeneralizationTRJ_0_1, data.mGeneralizationTRJ_1_10, data.mGeneralizationTRJ_10_100);
    data.mGeneralizationACC = cat(1, data.mGeneralizationACC_0_1, data.mGeneralizationACC_1_10, data.mGeneralizationACC_10_100);
    data.mGeneralizationPOT = cat(1, data.mGeneralizationPOT_0_1, data.mGeneralizationPOT_1_10, data.mGeneralizationPOT_10_100);

    names = fieldnames(data);
    for i = 1:numel(names)
        data.(names{i}) = dlarray(data.(names{i}), 'BC');
    end
end