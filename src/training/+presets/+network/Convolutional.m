function net = Convolutional(params)
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
    layersNN     = [layersNN, fullyConnectedLayer(128,"Name", sprintf("fc1"))];
    layersNN     = [layersNN, functionLayer(@reshape1DtoConv, "Name",  sprintf("rs1"), 'Formattable', true)];

    for i = 1:depthNN - 1
        layersNN     = [layersNN, convolution1dLayer(3, 64, "Name", sprintf("cv%d", i)), geluLayer("Name", sprintf("act%d", i))];
    end
    layersNN     = [layersNN, convolution1dLayer(3, 1, "Name", sprintf("cv%d", depthNN), 'Padding', 'same'), geluLayer("Name", sprintf("act%d", depthNN))];
    layersNN     = [layersNN, functionLayer(@reshapeBackFromConv, "Name",  sprintf("rs2"), 'Formattable', true)];
    layersNN     = [layersNN, fullyConnectedLayer(1 , "Name", sprintf("fc%d", depthNN), "WeightsInitializer", "zeros")];
    

    net = imagePretrainedNetwork("resnet18",Weights="none");
    net = removeLayers(net, "prob");
    layersNN = [];
    layersNN     = [layersNN, functionLayer(@reshapeBackFromConv, "Name",  sprintf("rs2"), 'Formattable', true)];
    layersNN     = [layersNN, fullyConnectedLayer(1, "Name", sprintf("fc7"))];
    net = replaceLayer(net, "fc1000", layersNN);

    layersFeatureEngineering = [ ...
        featureInputLayer(3, "Name", "featureinput")
        presets.layer.cart2sphLayer( ...
            "Name", "cart2sphLayer", "InputName", "Trajectory", "OutputNames", ["Spherical", "Radius"] ...
        )
        fullyConnectedLayer(64, "Name", sprintf("fc1"))
    ];
    layersFeatureEngineering = [layersFeatureEngineering; functionLayer(@reshape1DtoConv, "Name",  sprintf("rs1"), 'Formattable', true)];
    net = replaceLayer(net, "data", layersFeatureEngineering);
    
    net = replaceLayer(net, "conv1", convolution1dLayer(7, 64, "Name", sprintf("conv1"), 'Padding', 'same'));
    net = replaceLayer(net, "conv1_relu", geluLayer("Name", sprintf("gelu1")));
    net = replaceLayer(net, "pool1", maxPooling1dLayer(3, "Name", sprintf("pool1")));
    i = 2;
    while i ~= 6
        a = 'a';
        b = 'b';
        net = replaceLayer(net, sprintf("res%d%c_branch2%c", i,a,a), convolution1dLayer(3, 64, "Name", sprintf("res%d%c_branch2%c", i,a,a), 'Padding', 'same'));
        net = replaceLayer(net, sprintf("res%d%c_branch2%c", i,a,b), convolution1dLayer(3, 64, "Name", sprintf("res%d%c_branch2%c", i,a,b), 'Padding', 'same'));
        net = replaceLayer(net, sprintf("res%d%c_branch2%c", i,b,a), convolution1dLayer(3, 64, "Name", sprintf("res%d%c_branch2%c", i,b,a), 'Padding', 'same'));
        net = replaceLayer(net, sprintf("res%d%c_branch2%c", i,b,b), convolution1dLayer(3, 64, "Name", sprintf("res%d%c_branch2%c", i,b,b), 'Padding', 'same'));
        net = replaceLayer(net, sprintf("bn%d%c_branch2%c", i,a,a), batchNormalizationLayer("Name", sprintf("bn%d%c_branch2%c", i,a,a)));
        net = replaceLayer(net, sprintf("bn%d%c_branch2%c", i,a,b), batchNormalizationLayer("Name", sprintf("bn%d%c_branch2%c", i,a,b)));
        net = replaceLayer(net, sprintf("bn%d%c_branch2%c", i,b,a), batchNormalizationLayer("Name", sprintf("bn%d%c_branch2%c", i,b,a)));
        net = replaceLayer(net, sprintf("bn%d%c_branch2%c", i,b,b), batchNormalizationLayer("Name", sprintf("bn%d%c_branch2%c", i,b,b)));
        net = replaceLayer(net, sprintf("res%d%c_branch2%c_relu", i,a,a), geluLayer("Name", sprintf("res%d%c_branch2%c_gelu", i,a,a)));
        net = replaceLayer(net, sprintf("res%d%c_relu", i,a), geluLayer("Name", sprintf("res%d%c_gelu", i,a)));
        net = replaceLayer(net, sprintf("res%d%c_branch2%c_relu", i,b,a), geluLayer("Name", sprintf("bn%d%c_branch2%c_gelu", i,b,a)));
        net = replaceLayer(net, sprintf("res%d%c_relu", i,b), geluLayer("Name", sprintf("res%d%c_gelu", i,b)));
        i = i + 1;
    end
    net = replaceLayer(net, sprintf("res3a_branch1"), convolution1dLayer(3, 64, "Name", sprintf("res3a_branch1"), 'Padding', 'same'));
    net = replaceLayer(net, sprintf("res4a_branch1"), convolution1dLayer(3, 64, "Name", sprintf("res4a_branch1"), 'Padding', 'same'));
    net = replaceLayer(net, sprintf("res5a_branch1"), convolution1dLayer(3, 64, "Name", sprintf("res5a_branch1"), 'Padding', 'same'));
    net = replaceLayer(net, sprintf("bn3a_branch1"), batchNormalizationLayer("Name", sprintf("bn3a_branch1")));
    net = replaceLayer(net, sprintf("bn4a_branch1"), batchNormalizationLayer("Name", sprintf("bn4a_branch1")));
    net = replaceLayer(net, sprintf("bn5a_branch1"), batchNormalizationLayer("Name", sprintf("bn5a_branch1")));
    net = replaceLayer(net, "pool5", globalAveragePooling1dLayer("Name", sprintf("pool5")));

    % Add layers
    %net = addLayers(net, layersFeatureEngineering);
    %net = addLayers(net, layersNN);
    net = addLayers(net, presets.layer.scaleNNPotentialLayer( ...
        "Name", "scaleNNPotentialLayer"       , "InputNames", ["Potential", "Radius"]        , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.analyticModelLayer("mu", params.mu, ...
        "Name", "analyticModelLayer"          , "InputName" , "Radius"                       , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.fuseModelsLayer( ...
        "Name", "fuseModelsLayer"             , "InputNames", ["PotNN", "PotLF"]             , "OutputName", "Potential" ...
    ));
    net = addLayers(net, presets.layer.applyBoundaryConditionsLayer( ...
        "Name", "applyBoundaryConditionsLayer", "InputNames", ["PotFused", "PotLF", "Radius"], "OutputName", "Potential" ...
    ));

    % Connect NN layers
    %net = connectLayers(net, "cart2sphLayer/Spherical", "fc1/in"                         );
    net = connectLayers(net, "fc7"                    , "scaleNNPotentialLayer/Potential");
    net = connectLayers(net, "cart2sphLayer/Radius"   , "scaleNNPotentialLayer/Radius"   );

    % Connect Low-Fidelity Analytic Model layers
    net = connectLayers(net, "cart2sphLayer/Radius", "analyticModelLayer/Radius");

    % Connect Fusion layers
    net = connectLayers(net, "scaleNNPotentialLayer/Potential", "fuseModelsLayer/PotNN" );
    net = connectLayers(net, "analyticModelLayer/Potential"   , "fuseModelsLayer/PotLF" );

    % Connect Boundary Conditions layers
    net = connectLayers(net, "fuseModelsLayer/Potential"   , "applyBoundaryConditionsLayer/PotFused");
    net = connectLayers(net, "analyticModelLayer/Potential", "applyBoundaryConditionsLayer/PotLF"   );
    net = connectLayers(net, "cart2sphLayer/Radius"        , "applyBoundaryConditionsLayer/Radius"  );

    % Radius Identity Layer: output radius for loss component
    net = addLayers(net, identityLayer("Name", "RadiusIdentity"));
    net = connectLayers(net, "cart2sphLayer/Radius", "RadiusIdentity");




    function X = reshape1DtoConv(X)
        % Input X is [F × B] with dlarray labels 'CB'
        % Reshape to [F × 1 × B] and relabel as 'SCB'
    
        %[F, B] = size(X);
        %Y = reshape(X, [F, 1, B]);
        X = dlarray(X, 'SBC');  % S: Spatial, C: Channel, B: Batch
    end

    function X = reshapeBackFromConv(X)
        % Input is [F × 1 × B], output [F × B]
        X = dlarray(X, 'CSB');
        X = squeeze(X);  % Removes singleton dimension
        X = dlarray(X, 'CB');
  
    end

end