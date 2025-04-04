function data = PINN_GM_III(data)
    data  = pCombineMetricsRandomDist(data);
    data  = pAddExtraParameters(data);
    data  = pNonDimensionalize(data);
    data  = pMakeValidationSet(data);

    function data = pCombineMetricsRandomDist(data)
        % Handle 0R-1R 1R-10R 10R-100R data together
        data.mRandomTRJ = cat(1, data.mRandomTRJ_0_1, data.mRandomTRJ_1_10, data.mRandomTRJ_10_100);
        data.mRandomACC = cat(1, data.mRandomACC_0_1, data.mRandomACC_1_10, data.mRandomACC_10_100);
        data.mRandomPOT = cat(1, data.mRandomPOT_0_1, data.mRandomPOT_1_10, data.mRandomPOT_10_100);

        clearvars data.mRandomTRJ_0_1 data.mRandomTRJ_1_10 data.mRandomTRJ_10_100;
    end

    function data = pAddExtraParameters(data)
        % Radius Max
        data.params.rMax = 16000;   % Eros radius

        % Mu
        g              = 6.67430e-11;
        vol            = 2525994603183.156;   % from 8k file in dataset https://github.com/MartinAstro/GravNN
        density        = 2670;                % https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=eros
        data.params.mu = g * vol * density;

        % Eccentricity
        a             = data.params.rMax;
        b             = 3120;                      % Min Eros radius
        data.params.e = sqrt(1 - b ^ 2 / a ^ 2);
    end

    function data = pNonDimensionalize(data)
        % Non-dimensionalize the data
        % Values for the non-dimensionalization derived from Eros model
        sTRJ  = data.params.rMax;
        sPOT  = max(abs(data.tSurfacePOT));   % data.tRandomPOT NOT included as it has lower values than data.tSurfacePOT anyways
        sTIME = sqrt((sTRJ ^ 2) / sPOT);
        sACC  = sTRJ / (sTIME ^ 2);
        sMU   = sTIME ^ 2 / sTRJ ^ 3;

        % Scale training data
        data.tSurfaceTRJ = data.tSurfaceTRJ / sTRJ;
        data.tSurfaceACC = data.tSurfaceACC / sACC;
        data.tSurfacePOT = data.tSurfacePOT / sPOT;
        data.tRandomTRJ  = data.tRandomTRJ  / sTRJ;
        data.tRandomACC  = data.tRandomACC  / sACC;
        data.tRandomPOT  = data.tRandomPOT  / sPOT;

        % Scale metrics data
        data.mPlanesTRJ = data.mPlanesTRJ / sTRJ;
        data.mPlanesACC = data.mPlanesACC / sACC;
        data.mPlanesPOT = data.mPlanesPOT / sPOT;
        data.mRandomTRJ = data.mRandomTRJ / sTRJ;
        data.mRandomACC = data.mRandomACC / sACC;
        data.mRandomPOT = data.mRandomPOT / sPOT;

        % Scale parameters
        data.params.mu = data.params.mu * sMU;
    end

    function data = pMakeValidationSet(data)
        % Generate shuffle indexes
        pSurfaceIdx = randperm(size(data.tSurfaceTRJ, 1));
        pRandomIdx  = randperm(size(data.tRandomTRJ, 1));

        % Shuffle data
        data.tSurfaceTRJ       = data.tSurfaceTRJ(pSurfaceIdx, :);
        data.tSurfaceACC       = data.tSurfaceACC(pSurfaceIdx, :);
        data.tSurfacePOT       = data.tSurfacePOT(pSurfaceIdx, :);
        data.tRandomTRJ        = data.tRandomTRJ(pRandomIdx, :);
        data.tRandomACC        = data.tRandomACC(pRandomIdx, :);
        data.tRandomPOT        = data.tRandomPOT(pRandomIdx, :);

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

        clearvars data.tSurfaceTRJ data.tSurfaceACC data.tSurfacePOT data.tRandomTRJ data.tRandomACC data.tRandomPOT;
    end

end