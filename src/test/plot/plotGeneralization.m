function [] = plotGeneralization(GeneralizationRadius, GeneralizationMetric, GeneralizationRadiusPoly, GeneralizationPoly)
    % Generalization: mpeLoss vs. distance(R), convert mpeLoss in log scale
    figure;
    hold on;
    semilogy(extractdata(GeneralizationRadiusPoly), extractdata(GeneralizationPoly)  , '.', 'Color', [127 127 127] ./ 256, 'DisplayName', 'Polyhedral');
    semilogy(extractdata(GeneralizationRadius)    , extractdata(GeneralizationMetric), '.', 'Color', [142 27 29] ./ 256, 'DisplayName', 'Generalization');

    set(gca, 'YScale', 'log');
    xlim([0, 20]);
    ylim([1e-4 1e3]);
    grid on;
    xlabel('Distance (R)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Palatino Linotype');
    ylabel('Percent Error', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Palatino Linotype');
    legend('show'); 
end