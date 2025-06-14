function net = KKLatest(params)
    net = dlnetwork();

    % Feature Engineering
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer()
    ];
    net = addLayers(net, layersFeatureEngineering);

    % Learning Blocks
    layersResidual = [ ...
        presets.layer.fourierLayer(5, 32, 1)
        presets.layer.rbfLayer(32, 32)
        ...
        fullyConnectedLayer(32, "Name", "fc1")
        presets.layer.sineLayer(30,"Name", "act1")
        fullyConnectedLayer(32, "Name", "fc3")
        additionLayer(2, "Name", "addRBFto3")
        geluLayer("Name", "act3")
        ...
        fullyConnectedLayer(32, "Name", "fc4")
        presets.layer.sineLayer(0,"Name", "act4")
        fullyConnectedLayer(32, "Name", "fc5")
        additionLayer(2, "Name", "add3to5")
        geluLayer("Name", "act5")
        ...
        fullyConnectedLayer(1 , "Name", "fc6", "WeightsInitializer", "zeros")
    ];
    net = addLayers(net, layersResidual);
    net = connectLayers(net, "cart2sphLayer/Spherical", "fourierLayer"     );
    net = connectLayers(net, "rbfLayer"               , "addRBFto3/in2");
    net = connectLayers(net, "act3"                   , "add3to5/in2"  );

    % Postprocessing
    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "fc6"                 , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius", "scaleNNPotentialLayer/Radius"   );

    net = addLayers(net, presets.layer.analyticModelLayer("mu", params.mu));
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer/Radius");

    net = addLayers(net, presets.layer.fuseModelsLayer());
    net = connectLayers(net, "scaleNNPotentialLayer/Potential", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer/Potential"   , "fuseModelsLayer/PotLF");

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer());
    net = connectLayers(net, "fuseModelsLayer/Potential"      , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer/Potential"   , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2sphLayer/Radius"           , "applyBoundaryConditionsLayer/Radius"  );

    % Extra Output
    net = addLayers(net, identityLayer("Name", "RadiusIdentity"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusIdentity");
end