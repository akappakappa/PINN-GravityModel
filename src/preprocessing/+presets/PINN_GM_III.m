function [dataOut, split] = PINN_GM_III(dataIn, splitPercentages)
    [shuffle, divide, const, spherical] = deal(struct);

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
    
    % Non-dimensionalize data
    const.starTRJ = 16000 * 3;
    const.starPOT = max(max(divide.trainPOT), max(divide.validationPOT));
    const.starTIME = sqrt((const.starTRJ ^ 2) / const.starPOT);
    const.starACC = const.starTRJ / (const.starTIME ^ 2);
    divide.trainTRJ = divide.trainTRJ / const.starTRJ;
    divide.trainACC = divide.trainACC / const.starACC;
    divide.trainPOT = divide.trainPOT / const.starPOT;
    divide.validationTRJ = divide.validationTRJ / const.starTRJ;
    divide.validationACC = divide.validationACC / const.starACC;
    divide.validationPOT = divide.validationPOT / const.starPOT;
    divide.testTRJ = divide.testTRJ / const.starTRJ;
    divide.testACC = divide.testACC / const.starACC;
    divide.testPOT = divide.testPOT / const.starPOT;
    
    

    function [sphTRJ, sphACC, sphPOT] = carttosph(cartTRJ, cartACC, cartPOT)
        % r = vecnorm(divide.trainTRJ, 2, 2);
        % theta = atan2(y, x);
        % phi = atan2(sqrt((x .^ 2) + (y .^ 2)), z);

        % Change to spherical coordinates
        x = cartTRJ(:, 1);
        y = cartTRJ(:, 2);
        z = cartTRJ(:, 3);
        [theta, phi, r] = cart2sph(x, y, z);
        s = sin(x ./ r);
        t = sin(y ./ r);
        u = sin(z ./ r);
        ri = r;
        ri(ri >= 1) = 1;
        re = r;
        re(re <= 1) = 1;
        re(re > 1) = 1 ./ re(re > 1);
        sphTRJ = [ri, re, s, t, u];

        % Rotate Accelerations
        s_theta = sin(theta);
        c_theta = cos(theta);
        s_phi = sin(phi);
        c_phi = cos(phi);
        r_hat = [s_phi .* c_theta, s_phi .* s_theta, c_phi];
        theta_hat = [c_phi .* c_theta, c_phi .* s_theta, -s_phi];
        phi_hat = [-s_theta, c_theta, zeros(size(s_theta))];
        rotation = cat(3, r_hat, theta_hat, phi_hat);
        rotation = permute(rotation, [2, 3, 1]);
        sphACC = permute(cartACC, [2, 3, 1]);
        sphACC = pagemtimes(rotation, sphACC);
        sphACC = permute(sphACC, [3, 1, 2]);

        sphPOT = cartPOT;

    end
    
    [spherical.trainTRJ, spherical.trainACC, spherical.trainPOT] = carttosph(divide.trainTRJ, divide.trainACC, divide.trainPOT);
    [spherical.validationTRJ, spherical.validationACC, spherical.validationPOT] = carttosph(divide.validationTRJ, divide.validationACC, divide.validationPOT);
    [spherical.testTRJ, spherical.testACC, spherical.testPOT] = carttosph(divide.testTRJ, divide.testACC, divide.testPOT);
    
    % Change to spherical coordinates
    %r = vecnorm(divide.trainTRJ, 2, 2);
    %x = divide.trainTRJ(:, 1);
    %y = divide.trainTRJ(:, 2);
    %z = divide.trainTRJ(:, 3);
    %s = sin(x ./ r);
    %t = sin(y ./ r);
    %u = sin(z ./ r);
    %ri = r;
    %ri(ri >= 1) = 1;
    %re = r;
    %re(re <= 1) = 1;
    %re(re > 1) = 1 ./ re(re > 1);
    %spherical.trainTRJ = [ri, re, s, t, u];
    
    % Rotate Accelerations
    %[theta,phi,r] = cart2sph(x,y,z);
    %theta = atan2(y, x);
    %phi = atan2(sqrt((x .^ 2) + (y .^ 2)), z);
    %s_theta = sin(theta);
    %c_theta = cos(theta);
    %s_phi = sin(phi);
    %c_phi = cos(phi);
    %r_hat = [s_phi .* c_theta, s_phi .* s_theta, c_phi];
    %theta_hat = [c_phi .* c_theta, c_phi .* s_theta, -s_phi];
    %phi_hat = [-s_theta, c_theta, zeros(size(s_theta))];
    %spherical.rotation = cat(3, r_hat, theta_hat, phi_hat);
    %spherical.rotation = permute(spherical.rotation, [2, 3, 1]);
    %spherical.trainACC = permute(divide.trainACC, [2, 3, 1]);
    %spherical.trainACC = pagemtimes(spherical.rotation, spherical.trainACC);
    %spherical.trainACC = permute(spherical.trainACC, [3, 1, 2]);

    clear divide;

    dataOut = spherical;
    split   = [size(dataOut.trainTRJ, 1), size(dataOut.validationTRJ, 1), size(dataOut.testTRJ, 1)];

    clear spherical;
end