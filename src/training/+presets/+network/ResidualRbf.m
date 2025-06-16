function net = ResidualRbf(params)
    net = dlnetwork();

    % Feature Engineering
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer()
    ];
    net = addLayers(net, layersFeatureEngineering);

    % Learning
    layersNN = [ ...
        presets.layer.rbfLayer(5, 32)
        ...
        fullyConnectedLayer(32, "Name", "fc2")
        geluLayer("Name", "act2")
        fullyConnectedLayer(32, "Name", "fc3")
        geluLayer("Name", "act3")
        fullyConnectedLayer(32, "Name", "fc4")
        additionLayer(2, "Name", "add1to4")
        geluLayer("Name", "act4")
        ...
        fullyConnectedLayer(32, "Name", "fc5")
        geluLayer("Name", "act5")
        fullyConnectedLayer(32, "Name", "fc6")
        additionLayer(2, "Name", "add4to6")
        geluLayer("Name", "act6")
        ...
        fullyConnectedLayer(1 , "Name", "fcfinal", "WeightsInitializer", "zeros")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2sphLayer/Spherical", "rbfLayer");
    net = connectLayers(net, "rbfLayer", "add1to4/in2");
    net = connectLayers(net, "act4"    , "add4to6/in2");

    % Postprocessing
    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "fcfinal"             , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius", "scaleNNPotentialLayer/Radius"   );

    net = addLayers(net, presets.layer.analyticModelLayer("mu", params.mu));
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer");

    net = addLayers(net, presets.layer.fuseModelsLayer());
    net = connectLayers(net, "scaleNNPotentialLayer", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer"   , "fuseModelsLayer/PotLF");

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer());
    net = connectLayers(net, "fuseModelsLayer"      , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer"   , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2sphLayer/Radius" , "applyBoundaryConditionsLayer/Radius"  );

    % Extra Output
    net = addLayers(net, identityLayer("Name", "RadiusOutput"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusOutput");
end