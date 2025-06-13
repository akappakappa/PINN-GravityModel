function net = Convolutional(params)
    net = dlnetwork();

    % Define the feature engineering layers
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer( ...
            "Name", "cart2sphLayer", "InputName", "Trajectory", "OutputNames", ["Spherical", "Radius"] ...
        )
    ];

    % Define the NN layers: 6 * (FullyConnected + GELU) + FullyConnected
    layersNN = [];
    depthNN  = 7;
    layersNN     = [layersNN, fullyConnectedLayer(128,"Name", sprintf("fc1"))];
    layersNN     = [layersNN, functionLayer(@reshape1DtoConv, "Name",  sprintf("rs1"), 'Formattable', true)];

    for i = 1:depthNN - 1
        layersNN     = [layersNN, convolution1dLayer(3, 64, "Name", sprintf("cv%d", i)), geluLayer("Name", sprintf("act%d", i))];
    end
    layersNN     = [layersNN, convolution1dLayer(3, 1, "Name", sprintf("cv%d", depthNN)), geluLayer("Name", sprintf("act%d", depthNN))];
    layersNN     = [layersNN, functionLayer(@reshapeBackFromConv, "Name",  sprintf("rs2"), 'Formattable', true)];
    layersNN     = [layersNN, fullyConnectedLayer(1 , "Name", sprintf("fc%d", depthNN), "WeightsInitializer", "zeros")];
    
    net = imagePretrainedNetwork("resnet18",Weights="none");
    

    % Add layers
    net = addLayers(net, layersFeatureEngineering);
    %net = addLayers(net, layersNN);
    net = addLayers(net, presets.layer.scaleNNPotentialLayer( ...
        "Name", "scaleNNPotentialLayer"       , "InputNames", ["Potential", "Radius"]        , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.analyticModelLayer("mu", params.mu, ...
        "Name", "analyticModelLayer"          , "InputName" , "Radius"                       , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.fuseModelsLayer( ...
        "Name", "fuseModelsLayer"             , "InputNames", ["PotNN", "PotLF"]             , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer( ...
        "Name", "applyBoundaryConditionsLayer", "InputNames", ["PotFused", "PotLF", "Radius"], "OutputName", "Potential" ...
    ));

    % Connect NN layers
    net = connectLayers(net, "cart2sphLayer/Spherical", "fc1/in"                         );
    net = connectLayers(net, "fc7"                    , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius"   , "scaleNNPotentialLayer/Radius"   );

    % Connect Low-Fidelity Analytic Model layers
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer/Radius");

    % Connect Fusion layers
    net = connectLayers(net, "scaleNNPotentialLayer/Potential", "fuseModelsLayer/PotNN" );
    net = connectLayers(net, "analyticModelLayer/Potential"   , "fuseModelsLayer/PotLF" );

    % Connect Boundary Conditions layers
    net = connectLayers(net, "fuseModelsLayer/Potential"   , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer/Potential", "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2sphLayer/Radius"        , "applyBoundaryConditionsLayer/Radius"  );

    % Radius Identity Layer: output radius for loss component
    net = addLayers(net, identityLayer("Name", "RadiusIdentity"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusIdentity");




    function X = reshape1DtoConv(X)
        % Input X is [F × B] with dlarray labels 'CB'
        % Reshape to [F × 1 × B] and relabel as 'SCB'
    
        %[F, B] = size(X);
        %Y = reshape(X, [F, 1, B]);
        X = dlarray(X, 'SBC');  % S: Spatial, C: Channel, B: Batch
    end

    function X = reshapeBackFromConv(X)
        % Input is [F × 1 × B], output [F × B]
        X = squeeze(X);  % Removes singleton dimension
        X = dlarray(X, 'CB');
  
    end

end