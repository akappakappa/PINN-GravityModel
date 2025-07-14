close all; clear; clc;

if batchStartupOptionUsed
    return;
end
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

data   = mLoadData("./src/preprocessing/metricsData.mat");
nnameA = "net_PINN_GM_III";
nnameB = "net_f_FC_wr";
netA   = load("./src/training/" + nnameA + ".mat").net;
netB   = load("./src/training/" + nnameB + ".mat").net;
[netA_G, netA_Gr] = dlfeval(@presets.mpeLoss, netA, data.mGeneralizationTRJ, data.mGeneralizationACC, data.mGeneralizationPOT);
[netB_G, netB_Gr] = dlfeval(@presets.mpeLoss, netB, data.mGeneralizationTRJ, data.mGeneralizationACC, data.mGeneralizationPOT);
[poly_G, poly_Gr] = presets.comparePolyhedral(data.mGeneralizationTRJ, data.mGeneralizationACC, data.mGeneralizationPOT, data.pGeneralizationTRJ, data.pGeneralizationACC, data.pGeneralizationPOT);
figure;
hold on;
xregion(0 , 1  , "FaceColor", [ 99  99  99] ./ 255, "EdgeColor", [0.5 0.5 0.5], "DisplayName", "Interior"     );
xregion(1 , 10 , "FaceColor", [149 149 149] ./ 255, "EdgeColor", [0.5 0.5 0.5], "DisplayName", "Exterior"     );
xregion(10, 100, "FaceColor", [199 199 199] ./ 255, "EdgeColor", [0.5 0.5 0.5], "DisplayName", "Extrapolation"); 
[labelA, labelB] = deal(strrep(sprintf('Generalization %s', nnameA), '_', '\_'), strrep(sprintf('Generalization %s', nnameB), '_', '\_'));
semilogy(extractdata(poly_Gr), extractdata(poly_G), '.', "MarkerSize", 5, "Color", [ 99 149 224] ./ 255, "DisplayName", "Polyhedral");
semilogy(extractdata(netA_Gr), extractdata(netA_G), '.', "MarkerSize", 5, "Color", [224  49  49] ./ 255, "DisplayName", labelA      );
semilogy(extractdata(netB_Gr), extractdata(netB_G), '.', "MarkerSize", 6, "Color", [49  149  49] ./ 255, "DisplayName", labelB      );
set(gca, "YScale", "log");
xlim([0 20]);
ylim([1e-4 1e3]);
grid on;
xlabel("Distance (R)" , "FontSize", 12, "FontWeight", "bold", "FontName", "Palatino Linotype");
ylabel("Percent Error", "FontSize", 12, "FontWeight", "bold", "FontName", "Palatino Linotype");
legend("show");
exportgraphics(gcf, "./utils/" + nnameA + "_vs_" + nnameB + ".png", "Resolution", 300);

function data = mLoadData(path)
    data = load(path);
    data.mGeneralizationTRJ = cat(1, data.mGeneralizationTRJ_0_1, data.mGeneralizationTRJ_1_10, data.mGeneralizationTRJ_10_100);
    data.mGeneralizationACC = cat(1, data.mGeneralizationACC_0_1, data.mGeneralizationACC_1_10, data.mGeneralizationACC_10_100);
    data.mGeneralizationPOT = cat(1, data.mGeneralizationPOT_0_1, data.mGeneralizationPOT_1_10, data.mGeneralizationPOT_10_100);
    data.pGeneralizationTRJ = cat(1, data.pGeneralizationTRJ_0_1, data.pGeneralizationTRJ_1_10, data.pGeneralizationTRJ_10_100);
    data.pGeneralizationACC = cat(1, data.pGeneralizationACC_0_1, data.pGeneralizationACC_1_10, data.pGeneralizationACC_10_100);
    data.pGeneralizationPOT = cat(1, data.pGeneralizationPOT_0_1, data.pGeneralizationPOT_1_10, data.pGeneralizationPOT_10_100);

    names = fieldnames(data);
    for i = 1:numel(names)
        data.(names{i}) = dlarray(data.(names{i}), 'BC');
    end
end