function net = PINN_GM_III(mu, e)
    net = dlnetwork();

    % Define the feature engineering layers
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        functionLayer(@presets.network.customLayer.cart2sphLayer, ...
            "Name", "cart2sphLayer", "InputName", "Trajectory", "OutputNames", ["Spherical", "Radius"], "Acceleratable", true ...
        )
    ];

    % Define the NN layers
    layersNN = [];
    depthNN  = 7;
    for i = 1:depthNN - 1
        layersNN = [layersNN, fullyConnectedLayer(32, "Name", sprintf("fc%d", i)), geluLayer("Name", sprintf("act%d", i))];
    end
    layersNN     = [layersNN, fullyConnectedLayer(1 , "Name", sprintf("fc%d", depthNN))];

    % Add layers
    net = addLayers(net, layersFeatureEngineering);
    net = addLayers(net, layersNN);
    net = addLayers(net, functionLayer(@presets.network.customLayer.scaleNNPotentialLayer, ...
        "Name", "scaleNNPotentialLayer"       , "InputNames", ["Potential", "Radius"]        , "OutputName", "Potential", "Acceleratable", true ...
    ));
    net = addLayers(net, functionLayer(@(Radius) presets.network.customLayer.analyticModelLayer(Radius, mu), ...
        "Name", "analyticModelLayer"          , "InputName" , "Radius"                       , "OutputName", "Potential", "Acceleratable", true ...
    ));
    net = addLayers(net, functionLayer(@(PotNN, PotLF, Radius) presets.network.customLayer.fuseModelsLayer(PotNN, PotLF, Radius, e), ...
        "Name", "fuseModelsLayer"             , "InputNames", ["PotNN", "PotLF", "Radius"]   , "OutputName", "Potential", "Acceleratable", true ...
    ));
    net = addLayers(net, functionLayer(@presets.network.customLayer.applyBoundaryConditionsLayer, ...
        "Name", "applyBoundaryConditionsLayer", "InputNames", ["PotFused", "PotLF", "Radius"], "OutputName", "Potential", "Acceleratable", true ...
    ));

    % Connect NN layers
    net = connectLayers(net, "cart2sphLayer/Spherical", "fc1/in"                         );
    net = connectLayers(net, "fc7"                    , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius"   , "scaleNNPotentialLayer/Radius"   );

    % Connect Low-Fidelity Analytic Model layers
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer/Radius");

    % Connect Fusion layers
    net = connectLayers(net, "scaleNNPotentialLayer/Potential", "fuseModelsLayer/PotNN" );
    net = connectLayers(net, "analyticModelLayer/Potential"   , "fuseModelsLayer/PotLF" );
    net = connectLayers(net, "cart2sphLayer/Radius"           , "fuseModelsLayer/Radius");

    % Connect Boundary Conditions layers
    net = connectLayers(net, "fuseModelsLayer/Potential"   , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer/Potential", "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2sphLayer/Radius"        , "applyBoundaryConditionsLayer/Radius"  );
end