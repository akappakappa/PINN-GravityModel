function gradients = thresholdL2Norm(gradients, threshold)
    % THRESHOLDL2NORM  Thresholds the L2 norm of the gradients.
    %   GRADIENTS = THRESHOLDL2NORM(GRADIENTS, THRESHOLD) thresholds the L2 norm of the gradients to the specified threshold.
    
    l2Norm = sqrt(sum(gradients(:) .^ 2));
    if l2Norm > threshold
        gradients = gradients * (threshold / l2Norm);
    end
end