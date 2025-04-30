function net = PINN_GM_III_S(mu, e)
    net = dlnetwork();

    % Define the feature engineering layers
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer( ...
            "Name", "cart2sphLayer", "InputName", "Trajectory", "OutputNames", ["Spherical", "Radius"] ...
        )
    ];

    % Define the NN layers
    layersNN = [];
    depthNN  = 6;
    layersNN = [layersNN, presets.layer.sirenLayer(5, 32, 1, "Name", "siren1")];
    for i = 1:depthNN - 1
        layersNN = [layersNN, presets.layer.sirenLayer(32, 32, 1, "Name", sprintf("siren%d", i + 1))];
    end
    layersNN     = [layersNN, fullyConnectedLayer(1 , "Name", sprintf("fc%d", depthNN + 1))];

    % Add layers
    net = addLayers(net, layersFeatureEngineering);
    net = addLayers(net, layersNN);
    net = addLayers(net, presets.layer.scaleNNPotentialLayer( ...
        "Name", "scaleNNPotentialLayer"       , "InputNames", ["Potential", "Radius"]        , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.analyticModelLayer("mu", mu, ...
        "Name", "analyticModelLayer"          , "InputName" , "Radius"                       , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.fuseModelsLayer("e", e, ...
        "Name", "fuseModelsLayer"             , "InputNames", ["PotNN", "PotLF", "Radius"]   , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer( ...
        "Name", "applyBoundaryConditionsLayer", "InputNames", ["PotFused", "PotLF", "Radius"], "OutputName", "Potential" ...
    ));

    % Connect NN layers
    net = connectLayers(net, "cart2sphLayer/Spherical", "siren1/in"                         );
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

    % Radius Identity Layer 
    net = addLayers(net, identityLayer("Name", "RadiusIdentity"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusIdentity");
end