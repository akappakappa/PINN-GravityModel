function net = f_RES1(params)
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

        identityLayer("Name", "skip1")
        fullyConnectedLayer(32)
        additionLayer(2, "Name", "add1")
        geluLayer()

        identityLayer("Name", "skip2")
        fullyConnectedLayer(32)
        additionLayer(2, "Name", "add2")
        geluLayer()

        identityLayer("Name", "skip3")
        fullyConnectedLayer(32)
        additionLayer(2, "Name", "add3")
        geluLayer()
        
        identityLayer("Name", "skip4")
        fullyConnectedLayer(32)
        additionLayer(2, "Name", "add4")
        geluLayer()

        identityLayer("Name", "skip5")
        fullyConnectedLayer(32)
        additionLayer(2, "Name", "add5")
        geluLayer()

        fullyConnectedLayer(1, "WeightsInitializer", "zeros")
        ...
        identityLayer("Name", "nnout")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2SphLayer/Spherical", "nnin");
    net = connectLayers(net, "skip1", "add1/in2");
    net = connectLayers(net, "skip2", "add2/in2");
    net = connectLayers(net, "skip3", "add3/in2");
    net = connectLayers(net, "skip4", "add4/in2");
    net = connectLayers(net, "skip5", "add5/in2");

    % Posprocessing
    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "nnout"               , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2SphLayer/Radius", "scaleNNPotentialLayer/Radius"   );

    net = addLayers(net, presets.layer.analyticModelLayer(params.mu, "FadeIn", true));
    net = connectLayers(net, "cart2SphLayer/Radius", "analyticModelLayer/Radius");

    net = addLayers(net, presets.layer.fuseModelsLayer());
    net = connectLayers(net, "scaleNNPotentialLayer", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer"   , "fuseModelsLayer/PotLF");

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer());
    net = connectLayers(net, "fuseModelsLayer"     , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer"  , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2SphLayer/Radius", "applyBoundaryConditionsLayer/Radius"  );
    
    % Extra Output
    net = addLayers(net, identityLayer("Name", "RadiusOutput"));
    net = connectLayers(net, "cart2SphLayer/Radius", "RadiusOutput");
end