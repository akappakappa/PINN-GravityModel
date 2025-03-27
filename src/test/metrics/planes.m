classdef planes < deep.Metric

    properties
        % (Required) Metric name.
        Name

        % Declare public metric properties here.

        % Any code can access these properties. Include here any properties
        % that you want to access or edit outside of the class.
    end

    properties (Access = private)
        % (Optional) Metric properties.

        % Declare private metric properties here.

        % Only members of the defining class can access these properties.
        % Include here properties that you do not want to edit outside
        % the class.
    end

    methods

        function metric = planes(args)
            % metric = fprMetric creates an fprMetric metric object.

            % metric = fprMetric(Name=name,NetworkOutput="out1",Maximize=0)
            % also specifies the optional Name option. By default,
            % the metric name is "FPR". By default,
            % the NetworkOutput is [], which corresponds to using all of
            % the network outputs. Maximize is set to 0 as the optimal value
            % occurs when the FPR is minimized.

            arguments
                args.Name = "Planes"
                args.NetworkOutput = []
                args.Maximize = 0
            end

            % Set the metric name.
            metric.Name = args.Name;

            % To support this metric for use with multi-output networks, set
            % the network output.
            metric.NetworkOutput = args.NetworkOutput;

            % To support this metric for early stopping and returning the
            % best network, set the maximize property. 
            metric.Maximize = args.Maximize;
        end

        %function metric = initialize(metric,batchY,batchT)
            % (Optional) Initialize metric.
            %
            % Use this function to initialize variables and run validation
            % checks.
            %
            % Inputs:
            %           metric - Metric to initialize
            %           batchY - Mini-batch of predictions
            %           batchT - Mini-batch of targets
            %
            % Output:
            %           metric - Initialized metric
            %
            % For networks with multiple outputs, replace batchY with
            % batchY1,...,batchYN and batchT with batchT1,...,batchTN,
            % where N is the number of network outputs. To create a metric
            % that supports any number of network outputs, replace batchY
            % and batchT with varargin.

            % Define metric initialization function here.
        %end

        function metric = reset(metric)
            % Reset metric properties.
            %
            % Use this function to reset the metric properties between
            % iterations.
            %
            % Input:
            %           metric - Metric containing properties to reset
            %
            % Output:
            %           metric - Metric with reset properties

            % Define metric reset function here.
            metric.Value = 0;
        end

        function metric = update(metric,batchY,batchT)
            % Update metric properties.
            %
            % Use this function to update metric properties that you use to
            % compute the final metric value.
            %
            % Inputs:
            %           metric - Metric containing properties to update
            %           batchY - Mini-batch of predictions
            %           batchT - Mini-batch of targets
            %
            % Output:
            %           metric - Metric with updated properties
            %
            % For networks with multiple outputs, replace batchY with
            % batchY1,...,batchYN and batchT with batchT1,...,batchTN,
            % where N is the number of network outputs. To create a metric
            % that supports any number of network outputs, replace batchY
            % and batchT with varargin.

            % Define metric update function here.
            diff = batchT - batchY;
            p = vecnorm(diff) / vecnorm(batchT) * 100 / size(batchT);
            metric.Value = (metric.Value + p) / 2;
        end

        function metric = aggregate(metric,metric2)
            % Aggregate metric properties.
            %
            % Use this function to define how to aggregate properties from
            % multiple instances of the same metric object during parallel
            % training.
            %
            % Inputs:
            %           metric  - Metric containing properties to aggregate
            %           metric2 - Metric containing properties to aggregate
            %
            % Output:
            %           metric - Metric with aggregated properties
            %
            % Define metric aggregation function here.
            metric.Value = (metric.Value + metric2.Value) / 2;
        end

        function val = evaluate(metric)
            % Evaluate metric properties.
            %
            % Use this function to define how to use the metric properties
            % to compute the final metric value.
            %
            % Input:
            %           metric - Metric containing properties to use to
            %           evaluate the metric value
            %
            % Output:
            %           val - Evaluated metric value
            %
            % To return multiple metric values, replace val with val1,...
            % valN.

            % Define metric evaluation function here.
            val = metric.Value;
        end
    end
end