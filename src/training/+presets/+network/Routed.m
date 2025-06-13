function net = Routed(params)
    net = dlnetwork();

    % Feature Engineering
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.c2sRadiusLayer()
        presets.layer.routingRadiusLayer()
    ];
    net = addLayers(net, layersFeatureEngineering);

    % Route Higher
    layersHigher = [ ...
        presets.layer.dynamicFullyConnectedLayer(4 , 32, "Name", "fc1_higher")
        presets.layer.dynamicGeluLayer("Name", "act1_higher")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc2_higher")
        presets.layer.dynamicGeluLayer("Name", "act2_higher")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc3_higher")
        presets.layer.dynamicGeluLayer("Name", "act3_higher")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc4_higher")
        presets.layer.dynamicAddResidualLayer("Name", "add1to4_higher")
        presets.layer.dynamicGeluLayer("Name", "act4_higher")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc5_higher")
        presets.layer.dynamicGeluLayer("Name", "act5_higher")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc6_higher")
        presets.layer.dynamicAddResidualLayer("Name", "add4to6_higher")
        presets.layer.dynamicGeluLayer("Name", "act6_higher")
        presets.layer.dynamicFullyConnectedLayer(32, 1 , "Name", "fc7_higher", "WeightsInit", "zeros")
    ];
    net = addLayers(net, layersHigher);
    net = connectLayers(net, "routingRadiusLayer/RouteHigher", "fc1_higher/in");
    net = connectLayers(net, "act1_higher", "add1to4_higher/Residual");
    net = connectLayers(net, "act4_higher", "add4to6_higher/Residual");

    net = addLayers(net, presets.layer.scaleNNPotentialLayer());
    net = connectLayers(net, "fc7_higher"                     , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "routingRadiusLayer/RadiusHigher", "scaleNNPotentialLayer/Radius"   );

    net = addLayers(net, presets.layer.analyticModelLayer("mu", params.mu));
    net = connectLayers(net, "routingRadiusLayer/RadiusHigher", "analyticModelLayer/Radius");

    net = addLayers(net, presets.layer.fuseModelsLayer());
    net = connectLayers(net, "scaleNNPotentialLayer/Potential", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer/Potential"   , "fuseModelsLayer/PotLF");

    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer());
    net = connectLayers(net, "fuseModelsLayer/Potential"      , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer/Potential"   , "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "routingRadiusLayer/RadiusHigher", "applyBoundaryConditionsLayer/Radius"  );

    % Route Lower
    layersLower = [ ...
        presets.layer.dynamicFullyConnectedLayer(4 , 32, "Name", "fc1_lower")
        presets.layer.dynamicGeluLayer("Name", "act1_lower")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc2_lower")
        presets.layer.dynamicGeluLayer("Name", "act2_lower")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc3_lower")
        presets.layer.dynamicGeluLayer("Name", "act3_lower")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc4_lower")
        presets.layer.dynamicAddResidualLayer("Name", "add1to4_lower")
        presets.layer.dynamicGeluLayer("Name", "act4_lower")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc5_lower")
        presets.layer.dynamicGeluLayer("Name", "act5_lower")
        presets.layer.dynamicFullyConnectedLayer(32, 32, "Name", "fc6_lower")
        presets.layer.dynamicAddResidualLayer("Name", "add4to6_lower")
        presets.layer.dynamicGeluLayer("Name", "act6_lower")
        presets.layer.dynamicFullyConnectedLayer(32, 1 , "Name", "fc7_lower", "WeightsInit", "zeros")
    ];
    net = addLayers(net, layersLower);
    net = connectLayers(net, "routingRadiusLayer/RouteLower", "fc1_lower/in");
    net = connectLayers(net, "act1_lower", "add1to4_lower/Residual");
    net = connectLayers(net, "act4_lower", "add4to6_lower/Residual");

    % Merge Higher and Lower Routes
    net = addLayers(net, presets.layer.mergingRadiusLayer());
    net = connectLayers(net, "applyBoundaryConditionsLayer/Potential", "mergingRadiusLayer/PotHigher"   );
    net = connectLayers(net, "fc7_lower"                             , "mergingRadiusLayer/PotLower"    );
    net = connectLayers(net, "routingRadiusLayer/RouteIndexes"       , "mergingRadiusLayer/RouteIndexes");

    % Extra Output
    net = addLayers(net, identityLayer("Name", "RadiusIdentity"));
    net = connectLayers(net, "routingRadiusLayer/Radius", "RadiusIdentity");
end