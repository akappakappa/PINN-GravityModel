function net = f_GM3_sine(params)
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
        presets.layer.sineLayer("Name", "act1")
        identityLayer("Name", "skip")

        fullyConnectedLayer(32)
        presets.layer.sineLayer("Name", "act2")
        additionLayer(2, "Name", "add1")

        fullyConnectedLayer(32)
        presets.layer.sineLayer("Name", "act3")
        additionLayer(2, "Name", "add2")

        fullyConnectedLayer(32)
        presets.layer.sineLayer("Name", "act4")
        additionLayer(2, "Name", "add3")

        fullyConnectedLayer(32)
        presets.layer.sineLayer("Name", "act5")
        additionLayer(2, "Name", "add4")

        fullyConnectedLayer(32)
        presets.layer.sineLayer("Name", "act6")
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