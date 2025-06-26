function net = FAC20(params)
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
        presets.layer.factorizedLayer(32, 20, "Name", "fac1")
        geluLayer()
        identityLayer("Name", "skip")

        presets.layer.factorizedLayer(32, 20, "Name", "fac2")
        geluLayer()
        additionLayer(2, "Name", "add1")

        presets.layer.factorizedLayer(32, 20, "Name", "fac3")
        geluLayer()
        additionLayer(2, "Name", "add2")

        presets.layer.factorizedLayer(32, 20, "Name", "fac4")
        geluLayer()
        additionLayer(2, "Name", "add3")

        presets.layer.factorizedLayer(32, 20, "Name", "fac5")
        geluLayer()
        additionLayer(2, "Name", "add4")

        presets.layer.factorizedLayer(32, 20, "Name", "fac6")
        geluLayer()
        additionLayer(2, "Name", "add5")

        presets.layer.factorizedLayer(1, 1, "WeightsInitializer", "zeros")
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

    net = addLayers(net, presets.layer.analyticModelLayer(params.mu));
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