function layer = fnoLayer(Width, NumModes, args)
    % fnoLayer Fourier Neural Operator (FNO) layer
    % <a href="matlab: web('https://www.mathworks.com/help/deeplearning/ug/solve-pde-using-fourier-neural-operator.html')">Solve PDE Using Fourier Neural Operator</a>

    arguments
        Width
        NumModes
        args.Name = "fnoLayer"
    end

    net = dlnetwork();

    layers = [
        identityLayer("Name", "id")
        presets.layer.spectralConvolution1dLayer(Width, NumModes, "Name", "sconv")
        additionLayer(2, "Name", "add")
    ];

    net = addLayers(net, layers);
    net = addLayers(net, convolution1dLayer(1, Width, "Name", "conv"));

    net = connectLayers(net, "id"  , "conv"   );
    net = connectLayers(net, "conv", "add/in2");

    layer = networkLayer(net, "Name", args.Name);
end