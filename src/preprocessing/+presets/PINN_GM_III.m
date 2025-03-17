function [data, split] = PINN_GM_III(data, splitPercentages)
    data  = ...
        prp_cart2sph( ...
        prp_nonDimensionalize( ...
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

    function data = prp_nonDimensionalize(data)
        starTRJ  = 16000 * 3;
        starPOT  = max(max(data.trainPOT), max(data.validationPOT));
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

    function data = prp_cart2sph(data)
        [data.trainTRJ, data.trainACC, data.trainPOT]                = prp_cart2sph_single(data.trainTRJ, data.trainACC, data.trainPOT);
        [data.validationTRJ, data.validationACC, data.validationPOT] = prp_cart2sph_single(data.validationTRJ, data.validationACC, data.validationPOT);
        [data.testTRJ, data.testACC, data.testPOT]                   = prp_cart2sph_single(data.testTRJ, data.testACC, data.testPOT);

        function [TRJ, ACC, POT] = prp_cart2sph_single(TRJ, ACC, POT)
            % Trajectory (cartesian to spherical)
            x = TRJ(:, 1);
            y = TRJ(:, 2);
            z = TRJ(:, 3);
            [theta, phi, r] = cart2sph(x, y, z);
            s = sin(x ./ r);
            t = sin(y ./ r);
            u = sin(z ./ r);

            ri          = r;
            ri(ri >= 1) = 1;
            re          = r;
            re(re <= 1) = 1;
            re(re > 1)  = 1 ./ re(re > 1);

            TRJ = [ri, re, s, t, u];

            % Acceleration (rotate)
            s_theta = sin(theta);
            c_theta = cos(theta);
            s_phi   = sin(phi);
            c_phi   = cos(phi);

            r_hat     = [s_phi .* c_theta, s_phi .* s_theta, c_phi];
            theta_hat = [c_phi .* c_theta, c_phi .* s_theta, -s_phi];
            phi_hat   = [-s_theta, c_theta, zeros(size(s_theta))];

            rotation = cat(3, r_hat, theta_hat, phi_hat);
            rotation = permute(rotation, [2, 3, 1]);

            ACC = permute(ACC, [2, 3, 1]);
            ACC = pagemtimes(rotation, ACC);
            ACC = permute(ACC, [3, 1, 2]);

            % Potential (proxy)
            scaleFactor = r;
            scaleFactor(scaleFactor <= 1) = 1;
            POT = POT .* scaleFactor;
        end
    end

end