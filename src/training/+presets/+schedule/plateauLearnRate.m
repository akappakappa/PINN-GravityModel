classdef plateauLearnRate < deep.LearnRateSchedule
    % plateauLearnRate Plateau learning rate schedule.
    % Reduce the learning rate when a plateau in validation loss is detected.
    %
    % plateauLearnRate Properties:
    %    Patience           - Number of epochs with no improvement before reducing the learning rate
    %    PatienceNow        - Number of epochs with no improvement since last reduction
    %    BestValidationLoss - Best validation loss so far
    %    Decay              - Decay factor
    %    MinDelta           - Minimum change in validation loss to be considered an improvement
    %    MinLearnRate       - Minimum permitted learning rate
    %
    % plateauLearnRate Methods:
    %    update - Lower bounded scaling of the learnrate if the validation loss hasn't significantly improved for too long

    properties
        Patience
        PatienceNow
        BestValidationLoss
        Decay
        MinDelta
        MinLearnRate
    end

    methods
        function schedule = plateauLearnRate(Patience, args)
            arguments
                Patience          (1, 1) double {mustBePositive, mustBeInteger}
                args.Decay        (1, 1) double {mustBeInRange(args.Decay, 0, 1)} = 0.5
                args.MinDelta     (1, 1) double {mustBeNonnegative}               = 1e-3
                args.MinLearnRate (1, 1) double {mustBePositive}                  = 1e-6
            end

            schedule.Patience           = Patience;
            schedule.PatienceNow        = Patience;
            schedule.BestValidationLoss = Inf;
            schedule.Decay              = args.Decay;
            schedule.MinDelta           = args.MinDelta;
            schedule.MinLearnRate       = args.MinLearnRate;
        end
        
        function [schedule, learnRate] = update(schedule, learnRate, validationLoss)
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