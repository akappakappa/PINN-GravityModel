function net = FcSkips(params)
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
        ...
        fullyConnectedLayer(32, "Name", "fc2")
        additionLayer(2, "Name", "add12")
        geluLayer("Name", "act2")
        fullyConnectedLayer(32, "Name", "fc3")
        additionLayer(2, "Name", "add13")
        geluLayer("Name", "act3")
        fullyConnectedLayer(32, "Name", "fc4")
        additionLayer(2, "Name", "add14")
        geluLayer("Name", "act4")
        fullyConnectedLayer(32, "Name", "fc5")
        additionLayer(2, "Name", "add15")
        geluLayer("Name", "act5")
        fullyConnectedLayer(32, "Name", "fc6")
        additionLayer(2, "Name", "add16")
        geluLayer("Name", "act6")
        ...
        fullyConnectedLayer(1 , "Name", "fcfinal", "WeightsInitializer", "zeros")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2sphLayer/Spherical", "fc1");
    net = connectLayers(net, "act1", "add12/in2");
    net = connectLayers(net, "act1", "add13/in2");
    net = connectLayers(net, "act1", "add14/in2");
    net = connectLayers(net, "act1", "add15/in2");
    net = connectLayers(net, "act1", "add16/in2");

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