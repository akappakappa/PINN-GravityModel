function net = Factorized32(params)
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
        presets.layer.factorizedLayer(32, 32, "Name", "fac1")
        geluLayer()
        presets.layer.factorizedLayer(32, 32, "Name", "fac2")
        geluLayer()
        presets.layer.factorizedLayer(32, 32, "Name", "fac3")
        geluLayer()
        presets.layer.factorizedLayer(32, 32, "Name", "fac4")
        geluLayer()
        presets.layer.factorizedLayer(32, 32, "Name", "fac5")
        geluLayer()
        presets.layer.factorizedLayer(32, 32, "Name", "fac6")
        geluLayer()
        fullyConnectedLayer(1, "WeightsInitializer", "zeros")
        ...
        identityLayer("Name", "nnout")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2SphLayer/Spherical", "nnin");

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

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer());
    net = connectLayers(net, "fuseModelsLayer"     , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer"  , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2SphLayer/Radius", "applyBoundaryConditionsLayer/Radius"  );
    
    % Extra Output
    net = addLayers(net, identityLayer("Name", "RadiusOutput"));
    net = connectLayers(net, "cart2SphLayer/Radius", "RadiusOutput");
end