function [] = plotGeneralization(nname, GeneralizationRadius, GeneralizationMetric, GeneralizationRadiusPoly, GeneralizationPoly)
    % Generalization: mpeLoss vs. distance(R), convert mpeLoss in log scale
    
    figure;
    hold on;
    xregion(0 , 1  , "FaceColor", [ 99  99  99] ./ 255, "EdgeColor", [0.5 0.5 0.5], "DisplayName", "Interior"     );
    xregion(1 , 10 , "FaceColor", [149 149 149] ./ 255, "EdgeColor", [0.5 0.5 0.5], "DisplayName", "Exterior"     );
    xregion(10, 100, "FaceColor", [199 199 199] ./ 255, "EdgeColor", [0.5 0.5 0.5], "DisplayName", "Extrapolation"); 
    semilogy(extractdata(GeneralizationRadiusPoly), extractdata(GeneralizationPoly)  , '.', "MarkerSize", 6, "Color", [ 99 149 224] ./ 255, "DisplayName", "Polyhedral"    );
    semilogy(extractdata(GeneralizationRadius)    , extractdata(GeneralizationMetric), '.', "MarkerSize", 6, "Color", [224  49  49] ./ 255, "DisplayName", "Generalization");

    set(gca, "YScale", "log");
    xlim([0 20]);
    ylim([1e-4 1e3]);
    grid on;
    xlabel("Distance (R)" , "FontSize", 12, "FontWeight", "bold", "FontName", "Palatino Linotype");
    ylabel("Percent Error", "FontSize", 12, "FontWeight", "bold", "FontName", "Palatino Linotype");
    legend("show");

    exportgraphics(gcf, "../../fig/" + nname + "/GNR_" + nname + ".png", "Resolution", 300);
end