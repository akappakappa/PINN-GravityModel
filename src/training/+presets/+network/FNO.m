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
        identityLayer("Name", "nnin")
        fullyConnectedLayer(16)
        geluLayer()
        fullyConnectedLayer(16)
        geluLayer()
        ...
        %reshapeLayer([1, 5], "OperationDimension", "spatial-channel")
        functionLayer(@(X) dlarray(reshape(X, [1, 16, size(X, 2)]), "SCB"), "Acceleratable", true, "Formattable", true)
        presets.layer.fnoLayer(16, 8, "Name", "fno1")
        geluLayer()
        presets.layer.fnoLayer(16, 8, "Name", "fno2")
        geluLayer()
        %reshapeLayer(1, "OperationDimension", "spatial-channel")
        functionLayer(@(X) dlarray(reshape(X, [16, size(X, 3)]), "CB"), "Acceleratable", true, "Formattable", true)
        ...
        fullyConnectedLayer(16)
        geluLayer()
        fullyConnectedLayer(16)
        geluLayer()
        ...
        fullyConnectedLayer(1, "WeightsInitializer", "zeros")
        identityLayer("Name", "nnout")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2sphLayer/Spherical", "nnin");

    % Posprocessing
    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "nnout"               , "scaleNNPotentialLayer/Potential");
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