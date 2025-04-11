classdef (Abstract) AbstractLearnRate < deep.LearnRateSchedule
    methods (Abstract)
        condition = isNewBest(schedule, validationLoss);
    end
end