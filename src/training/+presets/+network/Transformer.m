function net = Transformer(params)
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
        ...
        selfAttentionLayer(2, 32, "Name", "attention")
        additionLayer(2, "Name", "add1A")
        layerNormalizationLayer("Name", "normA")
        ...
        fullyConnectedLayer(32, "Name", "fc2")
        geluLayer("Name", "act2")
        fullyConnectedLayer(32, "Name", "fc3")
        additionLayer(2, "Name", "addA3")
        layerNormalizationLayer("Name", "norm2")
        ...
        fullyConnectedLayer(1 , "Name", "fcfinal", "WeightsInitializer", "zeros")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2sphLayer/Spherical", "fc1");
    net = connectLayers(net, "fc1"  , "add1A/in2");
    net = connectLayers(net, "normA", "addA3/in2");

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