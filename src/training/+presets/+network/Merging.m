function net = Merging(params)
    net = dlnetwork();

    % Define the feature engineering layers
    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer( ...
            "Name", "cart2sphLayer", "InputName", "Trajectory", "OutputNames", ["Spherical", "Radius"] ...
        )
    ];

    % Define the NN layers: 6 * (FullyConnected + GELU) + FullyConnected
    layersNN = [];
    depthNN  = 7;
    for i = 1:depthNN - 1
        layersNN = [layersNN, fullyConnectedLayer(32, "Name", sprintf("fc%d", i)), geluLayer("Name", sprintf("act%d", i))];
    end
    layersNN     = [layersNN, fullyConnectedLayer(1 , "Name", sprintf("fc%d", depthNN), "WeightsInitializer", "zeros")];

    % Add layers
    net = addLayers(net, layersFeatureEngineering);
    net = addLayers(net, layersNN);
    net = addLayers(net, presets.layer.scaleNNPotentialLayer( ...
        "Name", "scaleNNPotentialLayer", "InputNames", ["Potential", "Radius"], "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.mergingLayer("mu", params.mu, ...
        "Name", "mergingLayer"         , "InputNames", ["Potential", "Radius"], "OutputName", "Potential" ...
    ));

    % Connect NN layers
    net = connectLayers(net, "cart2sphLayer/Spherical", "fc1/in"                         );
    net = connectLayers(net, "fc7"                    , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius"   , "scaleNNPotentialLayer/Radius"   );

    % Connect Merging Layer
    net = connectLayers(net, "scaleNNPotentialLayer/Potential", "mergingLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius"           , "mergingLayer/Radius"   );

    % Radius Identity Layer: output radius for loss component
    net = addLayers(net, identityLayer("Name", "RadiusIdentity"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusIdentity");
end