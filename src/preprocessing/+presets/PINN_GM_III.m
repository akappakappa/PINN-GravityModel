function data = PINN_GM_III(data)
    % PINN_GM_III  Preprocess raw data for Physics-Informed Neural Network model training.
    %   DATA = PINN_GM_III(DATA) preprocesses the raw data by adding extra physical parameters, non-dimensionalizing the data, and splitting it into training and validation sets.

    data = pAddExtraParameters(data);
    data = pNonDimensionalize(data);
    data = pMakeValidationSet(data);

    function data = pAddExtraParameters(data)
        % PADDEXTRAPARAMETERS  Add extra physical parameters to the data structure.
        %   DATA = PADDEXTRAPARAMETERS(DATA) adds rMax=16'000, mu=g*vol*density and e=sqrt(1 - b ^ 2 / a ^ 2).

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
        % PNONDIMENSIONALIZE  Non-dimensionalize the data.
        %   DATA = PNONDIMENSIONALIZE(DATA) scales the data to be non-dimensional by dividing it by specific values derived from physical Eros properties.
        
        % Values for the non-dimensionalization derived from Eros model
        sTRJ  = data.params.rMax;
        Ulf   = -data.params.mu ./ sqrt(                                 ...
            cat(1, data.tSurfaceTRJ(:, 1), data.tRandomTRJ(:, 1)) .^ 2 + ...
            cat(1, data.tSurfaceTRJ(:, 2), data.tRandomTRJ(:, 2)) .^ 2 + ...
            cat(1, data.tSurfaceTRJ(:, 3), data.tRandomTRJ(:, 3)) .^ 2   ...
        );
        sPOT  = max(abs(cat(1, data.tSurfacePOT, data.tRandomPOT) - Ulf));   % data.tRandomPOT NOT included as it has lower values than data.tSurfacePOT anyways
        sTIME = sqrt((sTRJ ^ 2) / sPOT);
        sACC  = sTRJ / (sTIME ^ 2);
        sMU   = sTIME ^ 2 / sTRJ ^ 3;

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

        data.mGeneralizationTRJ_0_1    = data.mGeneralizationTRJ_0_1    ./ sTRJ;
        data.mGeneralizationACC_0_1    = data.mGeneralizationACC_0_1    ./ sACC;
        data.mGeneralizationPOT_0_1    = data.mGeneralizationPOT_0_1    ./ sPOT;
        data.mGeneralizationTRJ_1_10   = data.mGeneralizationTRJ_1_10   ./ sTRJ;
        data.mGeneralizationACC_1_10   = data.mGeneralizationACC_1_10   ./ sACC;
        data.mGeneralizationPOT_1_10   = data.mGeneralizationPOT_1_10   ./ sPOT;
        data.mGeneralizationTRJ_10_100 = data.mGeneralizationTRJ_10_100 ./ sTRJ;
        data.mGeneralizationACC_10_100 = data.mGeneralizationACC_10_100 ./ sACC;
        data.mGeneralizationPOT_10_100 = data.mGeneralizationPOT_10_100 ./ sPOT;

        data.mSurfaceTRJ = data.mSurfaceTRJ ./ sTRJ;
        data.mSurfaceACC = data.mSurfaceACC ./ sACC;
        data.mSurfacePOT = data.mSurfacePOT ./ sPOT;

        % Scale parameters
        data.params.mu = data.params.mu .* sMU;
    end

    function data = pMakeValidationSet(data)
        % PMAKEVALIDATIONSET  Split the data into training and validation sets.
        %   DATA = PMAKEVALIDATIONSET(DATA) shuffles the data and splits it into training and validation sets based on the specified split percentage.

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

        clearvars data.tSurfaceTRJ data.tSurfaceACC data.tSurfacePOT data.tRandomTRJ data.tRandomACC data.tRandomPOT;
    end
end