function [dataOut, split] = PINN_GM_III(dataIn, splitPercentages)
    [shuffle, divide, minmax] = deal(struct);

    % Shuffle dataset
    surfaceNum = size(dataIn.surfaceTRJ, 1);
    randomdNum = size(dataIn.randomdTRJ, 1);

    surfaceIdx = randperm(surfaceNum);
    randomdIdx = randperm(randomdNum);

    shuffle.surfaceTRJ = dataIn.surfaceTRJ(surfaceIdx, :);
    shuffle.surfaceACC = dataIn.surfaceACC(surfaceIdx, :);
    shuffle.surfacePOT = dataIn.surfacePOT(surfaceIdx, :);
    shuffle.randomdTRJ = dataIn.randomdTRJ(randomdIdx, :);
    shuffle.randomdACC = dataIn.randomdACC(randomdIdx, :);
    shuffle.randomdPOT = dataIn.randomdPOT(randomdIdx, :);

    % Divide into training, validation and test sets, and merge surface + randomd
    tSurfaceDiv = floor(surfaceNum * splitPercentages(1));
    tRandomdDiv = floor(randomdNum * splitPercentages(1));
    vSurfaceDiv = tSurfaceDiv + floor(surfaceNum * splitPercentages(2));
    vRandomdDiv = tRandomdDiv + floor(randomdNum * splitPercentages(2));

    divide.trainTRJ      = cat(1, shuffle.surfaceTRJ(1              :tSurfaceDiv, :), shuffle.randomdTRJ(1              :tRandomdDiv, :));
    divide.trainACC      = cat(1, shuffle.surfaceACC(1              :tSurfaceDiv, :), shuffle.randomdACC(1              :tRandomdDiv, :));
    divide.trainPOT      = cat(1, shuffle.surfacePOT(1              :tSurfaceDiv, :), shuffle.randomdPOT(1              :tRandomdDiv, :));
    divide.validationTRJ = cat(1, shuffle.surfaceTRJ(tSurfaceDiv + 1:vSurfaceDiv, :), shuffle.randomdTRJ(tRandomdDiv + 1:vRandomdDiv, :));
    divide.validationACC = cat(1, shuffle.surfaceACC(tSurfaceDiv + 1:vSurfaceDiv, :), shuffle.randomdACC(tRandomdDiv + 1:vRandomdDiv, :));
    divide.validationPOT = cat(1, shuffle.surfacePOT(tSurfaceDiv + 1:vSurfaceDiv, :), shuffle.randomdPOT(tRandomdDiv + 1:vRandomdDiv, :));
    divide.testTRJ       = cat(1, shuffle.surfaceTRJ(vSurfaceDiv + 1:end        , :), shuffle.randomdTRJ(vRandomdDiv + 1:end        , :));
    divide.testACC       = cat(1, shuffle.surfaceACC(vSurfaceDiv + 1:end        , :), shuffle.randomdACC(vRandomdDiv + 1:end        , :));
    divide.testPOT       = cat(1, shuffle.surfacePOT(vSurfaceDiv + 1:end        , :), shuffle.randomdPOT(vRandomdDiv + 1:end        , :));

    clear shuffle;

    % Min-Max scaling
    mmMinT = min(divide.trainTRJ);
    mmMaxT = max(divide.trainTRJ);
    mmMinA = min(min(divide.trainACC));
    mmMaxA = max(max(divide.trainACC));
    mmMinP = min(divide.trainPOT);
    mmMaxP = max(divide.trainPOT);
    
    minmax.trainTRJ      = rescale(divide.trainTRJ     , "InputMax", mmMaxT, "InputMin", mmMinT);
    minmax.trainACC      = rescale(divide.trainACC     , "InputMax", mmMaxA, "InputMin", mmMinA);
    minmax.trainPOT      = rescale(divide.trainPOT     , "InputMax", mmMaxP, "InputMin", mmMinP);
    minmax.validationTRJ = rescale(divide.validationTRJ, "InputMax", mmMaxT, "InputMin", mmMinT);
    minmax.validationACC = rescale(divide.validationACC, "InputMax", mmMaxA, "InputMin", mmMinA);
    minmax.validationPOT = rescale(divide.validationPOT, "InputMax", mmMaxP, "InputMin", mmMinP);
    minmax.testTRJ       = rescale(divide.testTRJ      , "InputMax", mmMaxT, "InputMin", mmMinT);
    minmax.testACC       = rescale(divide.testACC      , "InputMax", mmMaxA, "InputMin", mmMinA);
    minmax.testPOT       = rescale(divide.testPOT      , "InputMax", mmMaxP, "InputMin", mmMinP);

    clear divide;

    dataOut = minmax;
    split   = [size(dataOut.trainTRJ, 1), size(dataOut.validationTRJ, 1), size(dataOut.testTRJ, 1)];

    clear minmax;
end