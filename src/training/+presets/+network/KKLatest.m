function net = KKLatest(params)
    net = dlnetwork();

    % Feature Engineering
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer()
    ];
    net = addLayers(net, layersFeatureEngineering);

    % Learning Layers
    layersNN = [
        fullyConnectedLayer(32, "Name", "enc1")
        geluLayer("Name", "actenc1")
        fullyConnectedLayer(32, "Name", "enc2")
        geluLayer("Name", "actenc2")
        ...
        fullyConnectedLayer(32, "Name", "fc1")
        geluLayer("Name", "act1")
        additionLayer(2, "Name", "add1")
        fullyConnectedLayer(32, "Name", "fc2")
        geluLayer("Name", "act2")
        additionLayer(2, "Name", "add2")
        fullyConnectedLayer(32, "Name", "fc3")
        geluLayer("Name", "act3")
        additionLayer(2, "Name", "add3")
        fullyConnectedLayer(32, "Name", "fc4")
        geluLayer("Name", "act4")
        additionLayer(2, "Name", "add4")
        fullyConnectedLayer(32, "Name", "fc5")
        geluLayer("Name", "act5")
        additionLayer(2, "Name", "add5")
        fullyConnectedLayer(32, "Name", "fc6")
        geluLayer("Name", "act6")
        additionLayer(2, "Name", "add6")
        ...
        fullyConnectedLayer(1, "Name", "final")
    ];
    net = addLayers(net, layersNN);
    net = connectLayers(net, "cart2sphLayer/Spherical", "enc1");
    net = connectLayers(net, "actenc2", "add1/in2");
    net = connectLayers(net, "actenc2", "add2/in2");
    net = connectLayers(net, "actenc2", "add3/in2");
    net = connectLayers(net, "actenc2", "add4/in2");
    net = connectLayers(net, "actenc2", "add5/in2");
    net = connectLayers(net, "actenc2", "add6/in2");

    % Postprocessing
    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "final"               , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius", "scaleNNPotentialLayer/Radius"   );

    net = addLayers(net, presets.layer.analyticModelLayer("mu", params.mu));
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer/Radius");

    net = addLayers(net, presets.layer.fuseModelsLayer());
    net = connectLayers(net, "scaleNNPotentialLayer", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer"   , "fuseModelsLayer/PotLF");

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer());
    net = connectLayers(net, "fuseModelsLayer"      , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer"   , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2sphLayer/Radius" , "applyBoundaryConditionsLayer/Radius"  );

    % Extra Outputs
    net = addLayers(net, identityLayer("Name", "RadiusIdentity"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusIdentity");

    if false
        error("KKLatest: WIP");
    end
end