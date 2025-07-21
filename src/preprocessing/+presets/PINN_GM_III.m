function data = PINN_GM_III(data)
    % Preprocess raw data for Physics-Informed Neural Network model training.

    data = pAddExtraParameters(data);
    data = pNonDimensionalize(data);
    data = pMakeValidationSet(data);
    data = rmfield(data, {                           ...
        'tSurfaceTRJ', 'tSurfaceACC', 'tSurfacePOT', ...
        'tRandomTRJ' , 'tRandomACC' , 'tRandomPOT'   ...
    });



    % --- HELPER FUNCTIONS ---
    function data = pAddExtraParameters(data)
        % Add extra physical parameters to the data structure: rMax=16'000, mu=g*vol*density.

        data.params.rMax = 16000;               % Eros max radius
        g                = 6.67430e-11;         % Eros G
        vol              = 2525994603183.156;   % from 8k file in dataset https://github.com/MartinAstro/GravNN
        density          = 2670;                % https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=eros
        data.params.mu   = g * vol * density;   % mu=G*M
    end
    function data = pNonDimensionalize(data)
        % Non-dimensionalize the data with x-star,a-star,u-star,t-star
        
        sTRJ  = data.params.rMax;
        Ulf   = -data.params.mu ./ sqrt( ...   % Quickly evaluate point-mass potential
            cat(1, data.tSurfaceTRJ(:, 1), data.tRandomTRJ(:, 1)) .^ 2 + ...
            cat(1, data.tSurfaceTRJ(:, 2), data.tRandomTRJ(:, 2)) .^ 2 + ...
            cat(1, data.tSurfaceTRJ(:, 3), data.tRandomTRJ(:, 3)) .^ 2   ...
        );
        sPOT  = max(abs(cat(1, data.tSurfacePOT, data.tRandomPOT) - Ulf));
        sTIME = sqrt((sTRJ ^ 2) / sPOT);
        sACC  = sTRJ / (sTIME ^ 2);
        sMU   = sTRJ * sPOT;   % Would be: sTRJ ^ 3 / sTIME ^ 2, but can be simplified to sTRJ * sPOT

        % Scale training data
        data.tSurfaceTRJ = data.tSurfaceTRJ ./ sTRJ;
        data.tSurfaceACC = data.tSurfaceACC ./ sACC;
        data.tSurfacePOT = data.tSurfacePOT ./ sPOT;

        data.tRandomTRJ = data.tRandomTRJ ./ sTRJ;
        data.tRandomACC = data.tRandomACC ./ sACC;
        data.tRandomPOT = data.tRandomPOT ./ sPOT;

        % Scale metrics data
        data.mPlanesTRJ = data.mPlanesTRJ ./ sTRJ;
        data.mPlanesACC = data.mPlanesACC ./ sACC;
        data.mPlanesPOT = data.mPlanesPOT ./ sPOT;

        data.pPlanesTRJ = data.pPlanesTRJ ./ sTRJ;
        data.pPlanesACC = data.pPlanesACC ./ sACC;
        data.pPlanesPOT = data.pPlanesPOT ./ sPOT;

        data.mGeneralizationTRJ_0_1    = data.mGeneralizationTRJ_0_1    ./ sTRJ;
        data.mGeneralizationACC_0_1    = data.mGeneralizationACC_0_1    ./ sACC;
        data.mGeneralizationPOT_0_1    = data.mGeneralizationPOT_0_1    ./ sPOT;
        data.mGeneralizationTRJ_1_10   = data.mGeneralizationTRJ_1_10   ./ sTRJ;
        data.mGeneralizationACC_1_10   = data.mGeneralizationACC_1_10   ./ sACC;
        data.mGeneralizationPOT_1_10   = data.mGeneralizationPOT_1_10   ./ sPOT;
        data.mGeneralizationTRJ_10_100 = data.mGeneralizationTRJ_10_100 ./ sTRJ;
        data.mGeneralizationACC_10_100 = data.mGeneralizationACC_10_100 ./ sACC;
        data.mGeneralizationPOT_10_100 = data.mGeneralizationPOT_10_100 ./ sPOT;

        data.pGeneralizationTRJ_0_1    = data.pGeneralizationTRJ_0_1    ./ sTRJ;
        data.pGeneralizationACC_0_1    = data.pGeneralizationACC_0_1    ./ sACC;
        data.pGeneralizationPOT_0_1    = data.pGeneralizationPOT_0_1    ./ sPOT;
        data.pGeneralizationTRJ_1_10   = data.pGeneralizationTRJ_1_10   ./ sTRJ;
        data.pGeneralizationACC_1_10   = data.pGeneralizationACC_1_10   ./ sACC;
        data.pGeneralizationPOT_1_10   = data.pGeneralizationPOT_1_10   ./ sPOT;
        data.pGeneralizationTRJ_10_100 = data.pGeneralizationTRJ_10_100 ./ sTRJ;
        data.pGeneralizationACC_10_100 = data.pGeneralizationACC_10_100 ./ sACC;
        data.pGeneralizationPOT_10_100 = data.pGeneralizationPOT_10_100 ./ sPOT;

        data.mSurfaceTRJ = data.mSurfaceTRJ ./ sTRJ;
        data.mSurfaceACC = data.mSurfaceACC ./ sACC;
        data.mSurfacePOT = data.mSurfacePOT ./ sPOT;

        data.pSurfaceTRJ = data.pSurfaceTRJ ./ sTRJ;
        data.pSurfaceACC = data.pSurfaceACC ./ sACC;
        data.pSurfacePOT = data.pSurfacePOT ./ sPOT;

        % Scale parameters
        data.params.mu = data.params.mu / sMU;
    end
    function data = pMakeValidationSet(data)
        % Split the data into training and validation sets, following specified percentage.

        % Generate shuffle indexes
        pSurfaceIdx = randperm(size(data.tSurfaceTRJ, 1));
        pRandomIdx  = randperm(size(data.tRandomTRJ, 1));

        % Shuffle data
        data.tSurfaceTRJ = data.tSurfaceTRJ(pSurfaceIdx, :);
        data.tSurfaceACC = data.tSurfaceACC(pSurfaceIdx, :);
        data.tSurfacePOT = data.tSurfacePOT(pSurfaceIdx, :);
        
        data.tRandomTRJ  = data.tRandomTRJ(pRandomIdx, :);
        data.tRandomACC  = data.tRandomACC(pRandomIdx, :);
        data.tRandomPOT  = data.tRandomPOT(pRandomIdx, :);

        % Generate split indexes
        pSurfaceDiv = floor(size(data.tSurfaceTRJ, 1) * data.params.splitPercentage);
        pRandomDiv  = floor(size(data.tRandomTRJ, 1)  * data.params.splitPercentage);

        % Split data into training and validation sets
        data.trainTRJ      = cat(1, data.tSurfaceTRJ(1              :pSurfaceDiv, :), data.tRandomTRJ(1              :pRandomDiv, :));
        data.trainACC      = cat(1, data.tSurfaceACC(1              :pSurfaceDiv, :), data.tRandomACC(1              :pRandomDiv, :));
        data.trainPOT      = cat(1, data.tSurfacePOT(1              :pSurfaceDiv, :), data.tRandomPOT(1              :pRandomDiv, :));
        data.validationTRJ = cat(1, data.tSurfaceTRJ(pSurfaceDiv + 1:end        , :), data.tRandomTRJ(pRandomDiv + 1:end        , :));
        data.validationACC = cat(1, data.tSurfaceACC(pSurfaceDiv + 1:end        , :), data.tRandomACC(pRandomDiv + 1:end        , :));
        data.validationPOT = cat(1, data.tSurfacePOT(pSurfaceDiv + 1:end        , :), data.tRandomPOT(pRandomDiv + 1:end        , :));

        % Save split amounts
        data.params.split = [size(data.trainTRJ, 1), size(data.validationTRJ, 1)];
    end
end