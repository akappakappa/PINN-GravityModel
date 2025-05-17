classdef plateauLearnRate < deep.LearnRateSchedule
    % plateauLearnRate Learning rate schedule.
    %  This schedule reduces the learning rate when a plateau in validation loss is detected.

    properties
        BestValidationLoss   % Best validation loss so far
        Decay                % Decay factor
        MinDelta             % Minimum change in validation loss to be considered an improvement
        MinLearnRate         % Minimum permitted learning rate
        Patience             % Number of epochs with no improvement before reducing the learning rate
        PatienceNow          % Number of epochs with no improvement since last reduction
    end

    methods
        function schedule = plateauLearnRate(args)
            arguments
                args.Decay        (1, 1) double {mustBeInRange(args.Decay, 0, 1)}
                args.MinDelta     (1, 1) double {mustBeNonnegative}
                args.MinLearnRate (1, 1) double {mustBePositive}
                args.Patience     (1, 1) double {mustBePositive, mustBeInteger}
            end
            % Construct the learning rate schedule, given DECAY, MINDLETA, MINLEARNRATE, and PATIENCE.

            schedule.BestValidationLoss = Inf;
            schedule.Decay              = args.Decay;
            schedule.MinDelta           = args.MinDelta;
            schedule.MinLearnRate       = args.MinLearnRate;
            schedule.Patience           = args.Patience;
            schedule.PatienceNow        = args.Patience;
        end
        
        function [schedule, learnRate] = update(schedule, learnRate, validationLoss)
            % Compute the new learning rate based on the VALIDATIONLOSS and the current LEARNRATE.

            if validationLoss < schedule.BestValidationLoss - schedule.MinDelta
                schedule.BestValidationLoss = validationLoss;
                schedule.PatienceNow        = schedule.Patience;
            else
                schedule.PatienceNow = schedule.PatienceNow - 1;
            end
            if schedule.PatienceNow <= 0
                schedule.PatienceNow = schedule.Patience;
                learnRate            = max(learnRate * schedule.Decay, schedule.MinLearnRate);
            end
        end
    end
end