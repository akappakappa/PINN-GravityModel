function val = planes(net, Trj, Acc)

    [PotPred, state] = forward(net, Trj);

    % State data generated in the cart2sphLayer
    RotationMatrix = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "RotationMatrix");
    RotationMatrix = RotationMatrix{1};
    Radius         = state.Value(state.Layer == "cart2sphLayer" & state.Parameter == "Radius");
    Radius         = Radius{1};

    % Preprocess Acceleration (rotate)
    %Acc = extractdata(Acc);
    %Acc = permute(Acc, [1, 3, 2]);
    %Acc = pagemtimes(RotationMatrix, Acc);
    %Acc = permute(Acc, [1, 3, 2]);
    %Acc = dlarray(Acc, 'CB');

    % Preprocess Potential (proxy)
    ScaleFactor                   = Radius;
    ScaleFactor(ScaleFactor <= 1) = 1;
    PotPred                       = PotPred ./ ScaleFactor;


    AccPred = -dlgradient(sum(PotPred, 'all'), Trj, EnableHigherDerivatives = true);
    
    diff = Acc - AccPred;
    p = vecnorm(diff) ./ vecnorm(Acc);
    val = sum(p) / size(p,2);

end