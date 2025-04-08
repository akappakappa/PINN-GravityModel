function net = PINN_GM_III(mu, e)
    net = dlnetwork();
    lPreprocessing = [ ...
        featureInputLayer(3, "Name", "featureinput")
        functionLayer(@presets.network.customLayer.cart2sphLayer, "Name", "cart2sphLayer", "InputName", "TRJ", "OutputNames", ["SPH", "Radius"], "Acceleratable", true)
    ];
    lNN = [ ...
        fullyConnectedLayer(32, "Name", "fc1")
        geluLayer("Name", "act1")
        fullyConnectedLayer(32, "Name", "fc2")
        geluLayer("Name", "act2")
        fullyConnectedLayer(32, "Name", "fc3")
        geluLayer("Name", "act3")
        fullyConnectedLayer(32, "Name", "fc4")
        geluLayer("Name", "act4")
        fullyConnectedLayer(32, "Name", "fc5")
        geluLayer("Name", "act5")
        fullyConnectedLayer(32, "Name", "fc6")
        geluLayer("Name", "act6")
        fullyConnectedLayer(1, "Name", "fc7")
    ];

    % Add layers
    net = addLayers(net, lPreprocessing);
    net = addLayers(net, lNN);
    net = addLayers(net, functionLayer(@presets.network.customLayer.scaleNNPotentialLayer, "Name", "scaleNNPotentialLayer", "InputNames", ["Potential", "Radius"], "OutputName", "Potential", "Acceleratable", true));
    net = addLayers(net, functionLayer(@(Radius) presets.network.customLayer.analyticModelLayer(Radius, mu), "Name", "analyticModelLayer", "InputName", "Radius", "OutputName", "Potential", "Acceleratable", true));
    net = addLayers(net, functionLayer(@(PotNN, PotLF, Radius) presets.network.customLayer.fuseModelsLayer(PotNN, PotLF, Radius, e), "Name", "fuseModelsLayer", "InputNames", ["PotNN", "PotLF", "Radius"], "OutputName", "Potential", "Acceleratable", true));
    net = addLayers(net, functionLayer(@presets.network.customLayer.applyBoundaryConditionsLayer, "Name", "applyBoundaryConditionsLayer", "InputNames", ["PotFused", "PotLF", "Radius"], "OutputName", "Potential", "Acceleratable", true));

    % Connect NN layers
    net = connectLayers(net, "cart2sphLayer/SPH", "fc1/in");
    net = connectLayers(net, "fc7", "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius", "scaleNNPotentialLayer/Radius");

    % Connect Low-Fidelity Analytic Model layers
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer/Radius");

    % Connect Fusion layers
    net = connectLayers(net, "scaleNNPotentialLayer/Potential", "fuseModelsLayer/PotNN");
    net = connectLayers(net, "analyticModelLayer/Potential", "fuseModelsLayer/PotLF");
    net = connectLayers(net, "cart2sphLayer/Radius", "fuseModelsLayer/Radius");

    % Connect Boundary Conditions layers
    net = connectLayers(net, "fuseModelsLayer/Potential", "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer/Potential", "applyBoundaryConditionsLayer/PotLF");
    net = connectLayers(net, "cart2sphLayer/Radius", "applyBoundaryConditionsLayer/Radius");
end