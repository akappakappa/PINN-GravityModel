function val = planes(Y,T)
% Evaluate custom metric.

% Inputs:
%          Y - Formatted dlarray of predictions
%          T - Formatted dlarray of targets
%
% Outputs:
%           val - Metric value
%
% Define the metric function here.
diff = T - Y;
p = vecnorm(diff) ./ vecnorm(T) * 100;
val = sum(p) / size(p);

end