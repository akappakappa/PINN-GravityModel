function [data, split] = PINN_GM_III(data, splitPercentages)
    data  = ...
        prp_nonDimensionalize( ...
        prp_fixNegativePotential( ...
        prp_divide( ...
        prp_shuffle(data), splitPercentages ...
    )));
    split = [ ...
        size(data.trainTRJ     , 1), ...
        size(data.validationTRJ, 1), ...
        size(data.testTRJ      , 1) ...
    ];

    % Preprocessing functions (local)

    function data = prp_shuffle(data)
        surfaceIdx = randperm(size(data.surfaceTRJ, 1));
        randomdIdx = randperm(size(data.randomdTRJ, 1));

        data.surfaceTRJ = data.surfaceTRJ(surfaceIdx, :);
        data.surfaceACC = data.surfaceACC(surfaceIdx, :);
        data.surfacePOT = data.surfacePOT(surfaceIdx, :);
        data.randomdTRJ = data.randomdTRJ(randomdIdx, :);
        data.randomdACC = data.randomdACC(randomdIdx, :);
        data.randomdPOT = data.randomdPOT(randomdIdx, :);
    end

    function data = prp_divide(data, splitPercentages)
        surfaceNum = size(data.surfaceTRJ, 1);
        randomdNum = size(data.randomdTRJ, 1);

        tSurfaceDiv = floor(surfaceNum * splitPercentages(1));
        tRandomdDiv = floor(randomdNum * splitPercentages(1));
        vSurfaceDiv = tSurfaceDiv + floor(surfaceNum * splitPercentages(2));
        vRandomdDiv = tRandomdDiv + floor(randomdNum * splitPercentages(2));

        div = struct;
        div.trainTRJ      = cat(1, data.surfaceTRJ(1              :tSurfaceDiv, :), data.randomdTRJ(1              :tRandomdDiv, :));
        div.trainACC      = cat(1, data.surfaceACC(1              :tSurfaceDiv, :), data.randomdACC(1              :tRandomdDiv, :));
        div.trainPOT      = cat(1, data.surfacePOT(1              :tSurfaceDiv, :), data.randomdPOT(1              :tRandomdDiv, :));
        div.validationTRJ = cat(1, data.surfaceTRJ(tSurfaceDiv + 1:vSurfaceDiv, :), data.randomdTRJ(tRandomdDiv + 1:vRandomdDiv, :));
        div.validationACC = cat(1, data.surfaceACC(tSurfaceDiv + 1:vSurfaceDiv, :), data.randomdACC(tRandomdDiv + 1:vRandomdDiv, :));
        div.validationPOT = cat(1, data.surfacePOT(tSurfaceDiv + 1:vSurfaceDiv, :), data.randomdPOT(tRandomdDiv + 1:vRandomdDiv, :));
        div.testTRJ       = cat(1, data.surfaceTRJ(vSurfaceDiv + 1:end        , :), data.randomdTRJ(vRandomdDiv + 1:end        , :));
        div.testACC       = cat(1, data.surfaceACC(vSurfaceDiv + 1:end        , :), data.randomdACC(vRandomdDiv + 1:end        , :));
        div.testPOT       = cat(1, data.surfacePOT(vSurfaceDiv + 1:end        , :), data.randomdPOT(vRandomdDiv + 1:end        , :));

        data = div;
    end

    function data = prp_fixNegativePotential(data)
        data.trainPOT      = -data.trainPOT;
        data.validationPOT = -data.validationPOT;
        data.testPOT       = -data.testPOT;
    end

    function data = prp_nonDimensionalize(data)
        starTRJ  = 16000 * 3;
        starPOT  = max(data.trainPOT);
        starTIME = sqrt((starTRJ ^ 2) / starPOT);
        starACC  = starTRJ / (starTIME ^ 2);

        data.trainTRJ      = data.trainTRJ      / starTRJ;
        data.trainACC      = data.trainACC      / starACC;
        data.trainPOT      = data.trainPOT      / starPOT;
        data.validationTRJ = data.validationTRJ / starTRJ;
        data.validationACC = data.validationACC / starACC;
        data.validationPOT = data.validationPOT / starPOT;
        data.testTRJ       = data.testTRJ       / starTRJ;
        data.testACC       = data.testACC       / starACC;
        data.testPOT       = data.testPOT       / starPOT;
    end

end