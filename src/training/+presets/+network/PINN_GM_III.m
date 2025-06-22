function net = PINN_GM_III(params)
    net = dlnetwork();

    % Feature Engineering
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2SphLayer()
    ];
    net = addLayers(net, layersFeatureEngineering);

    % Learning
    layersNN = [
        identityLayer("Name", "nnin")
        ...
        fullyConnectedLayer(32)
        geluLayer()
        identityLayer("Name", "skip")

        fullyConnectedLayer(32)
        geluLayer()
        additionLayer(2, "Name", "add1")

        fullyConnectedLayer(32)
        geluLayer()
        additionLayer(2, "Name", "add2")

        fullyConnectedLayer(32)
        geluLayer()
        additionLayer(2, "Name", "add3")

        fullyConnectedLayer(32)
        geluLayer()
        additionLayer(2, "Name", "add4")

        fullyConnectedLayer(32)
        geluLayer()
        additionLayer(2, "Name", "add5")

        fullyConnectedLayer(1, "WeightsInitializer", "zeros")
        ...
        identityLayer("Name", "nnout")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2SphLayer/Spherical", "nnin");
    net = connectLayers(net, "skip", "add1/in2");
    net = connectLayers(net, "skip", "add2/in2");
    net = connectLayers(net, "skip", "add3/in2");
    net = connectLayers(net, "skip", "add4/in2");
    net = connectLayers(net, "skip", "add5/in2");

    % Posprocessing
    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "nnout"                     , "scaleNNPotentialLayer/Potential"   );
    net = connectLayers(net, "cart2SphLayer/RadiusInvExt", "scaleNNPotentialLayer/RadiusInvExt");

    net = addLayers(net, presets.layer.analyticModelLayer(params.mu));
    net = connectLayers(net, "cart2SphLayer/Radius"      , "analyticModelLayer/Radius"      );
    net = connectLayers(net, "cart2SphLayer/RadiusInvExt", "analyticModelLayer/RadiusInvExt");

    net = addLayers(net, presets.layer.fuseModelsLayer());
    net = connectLayers(net, "scaleNNPotentialLayer", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer"   , "fuseModelsLayer/PotLF");

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer("Mode", "tanh"));
    net = connectLayers(net, "fuseModelsLayer"     , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer"  , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2SphLayer/Radius", "applyBoundaryConditionsLayer/Radius"  );
    
    % Extra Output
    net = addLayers(net, identityLayer("Name", "RadiusOutput"));
    net = connectLayers(net, "cart2SphLayer/Radius", "RadiusOutput");
end