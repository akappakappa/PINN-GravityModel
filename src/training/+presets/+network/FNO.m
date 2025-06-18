function net = FNO(params)
    net = dlnetwork();

    % Feature Engineering
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer()
    ];
    net = addLayers(net, layersFeatureEngineering);

    % Learning
    layersNN = [
        fullyConnectedLayer(32, "Name", "fc1")
        geluLayer("Name", "act1")
        fullyConnectedLayer(32, "Name", "fc2")
        geluLayer("Name", "act2")
        fullyConnectedLayer(32, "Name", "fc3")
        geluLayer("Name", "act3")
        fullyConnectedLayer(32, "Name", "fc4")
        geluLayer("Name", "act4")
        fullyConnectedLayer(32, "Name", "fc5")
        geluLayer("Name", "act5")
        fullyConnectedLayer(32, "Name", "fc6")
        geluLayer("Name", "act6")
        ...
        fullyConnectedLayer(1 , "Name", "fcfinal", "WeightsInitializer", "zeros")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2sphLayer/Spherical", "fc1");

    % Posprocessing
    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "fcfinal"             , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius", "scaleNNPotentialLayer/Radius"   );

    net = addLayers(net, presets.layer.analyticModelLayer("mu", params.mu));
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer");

    net = addLayers(net, presets.layer.fuseModelsLayer());
    net = connectLayers(net, "scaleNNPotentialLayer", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer"   , "fuseModelsLayer/PotLF");

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer());
    net = connectLayers(net, "fuseModelsLayer"     , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer"  , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2sphLayer/Radius", "applyBoundaryConditionsLayer/Radius"  );
    
    % Extra Output
    net = addLayers(net, identityLayer("Name", "RadiusOutput"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusOutput");
end