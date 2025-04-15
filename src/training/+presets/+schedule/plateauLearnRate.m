classdef plateauLearnRate < deep.LearnRateSchedule
    properties
        BestValidationLoss
        Decay
        MinDelta
        MinLearnRate
        Patience
        PatienceNow
    end

    methods
        function schedule = plateauLearnRate(args)
            arguments
                args.Decay        (1, 1) double {mustBeInRange(args.Decay, 0, 1)}
                args.MinDelta     (1, 1) double {mustBeNonnegative}
                args.MinLearnRate (1, 1) double {mustBePositive}
                args.Patience     (1, 1) double {mustBePositive, mustBeInteger}
            end
            schedule.BestValidationLoss = Inf;
            schedule.Decay              = args.Decay;
            schedule.MinDelta           = args.MinDelta;
            schedule.MinLearnRate       = args.MinLearnRate;
            schedule.Patience           = args.Patience;
            schedule.PatienceNow        = args.Patience;
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