function net = RbfTest(params)
    net = dlnetwork();

    % Feature Engineering
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer()
    ];
    net = addLayers(net, layersFeatureEngineering);

    % Residual Blocks
    layersResidual = [ ...
        presets.layer.rbfLayer(5, 32)
        fullyConnectedLayer(32, "Name", "fc1")
        geluLayer("Name", "act1")
        fullyConnectedLayer(32, "Name", "fc2")
        geluLayer("Name", "act2")
        fullyConnectedLayer(32, "Name", "fc4")
        additionLayer(2, "Name", "add1to4")
        geluLayer("Name", "act4")
        fullyConnectedLayer(32, "Name", "fc5")
        geluLayer("Name", "act5")
        fullyConnectedLayer(32, "Name", "fc6")
        additionLayer(2, "Name", "add4to6")
        geluLayer("Name", "act6")
        fullyConnectedLayer(1 , "Name", "fc7", "WeightsInitializer", "zeros")
    ];
    net = addLayers(net, layersResidual);
    net = connectLayers(net, "cart2sphLayer/Spherical", "rbfLayer");
    net = connectLayers(net, "rbfLayer", "add1to4/in2");
    net = connectLayers(net, "act4"    , "add4to6/in2");

    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "fc7"                 , "scaleNNPotentialLayer/Potential");
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