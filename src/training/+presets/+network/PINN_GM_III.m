function layers = PINN_GM_III(preprocessingLayer)
    layers = [ ...
        featureInputLayer(3, "Name", "featureinput")
        preprocessingLayer
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
        fullyConnectedLayer(1)
    ];
end