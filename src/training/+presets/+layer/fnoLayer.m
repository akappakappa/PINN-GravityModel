function layer = fnoLayer(width, nmodes, args)
    arguments
        width
        nmodes
        args.Name = "fnoLayer";
    end
    net = dlnetwork();
    layers = [
        identityLayer("Name", "id")
        presets.layer.spectralConvolution1dLayer(width, nmodes, "Name", "sconv")
        additionLayer(2, "Name", "add")
    ];
    net   = addLayers(net, layers);
    net   = addLayers(net, convolution1dLayer(1, 32, "Name", "conv"));
    net   = connectLayers(net, "id"  , "conv"   );
    net   = connectLayers(net, "conv", "add/in2");
    layer = networkLayer(net, "Name", args.Name);
end