function net = PINN_GM_I()
    % Define the layers
    layers = [featureInputLayer(3, "Name", "featureinput")];
    depthNN = 9;
    for i = 1:depthNN - 1
        layers = [layers, fullyConnectedLayer(20, "Name", sprintf("fc%d", i)), geluLayer("Name", sprintf("act%d", i))];
    end
    layers     = [layers, fullyConnectedLayer(1 , "Name", sprintf("fc%d", depthNN))];
    
    % Add layers
    net = dlnetwork(layers);
end