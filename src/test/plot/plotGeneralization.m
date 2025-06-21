function [] = plotGeneralization(GeneralizationRadius, GeneralizationMetric, GeneralizationRadiusPoly, GeneralizationPoly)
    % Generalization: mpeLoss vs. distance(R), convert mpeLoss in log scale
    figure;
    hold on;
    semilogy(extractdata(GeneralizationRadius)    , extractdata(GeneralizationMetric), '.', 'DisplayName', 'Generalization');
    semilogy(extractdata(GeneralizationRadiusPoly), extractdata(GeneralizationPoly)  , '.', 'DisplayName', 'Polyhedral');

    set(gca, 'YScale', 'log');
    xlim([0, 20]);
    grid on;
    xlabel('Distance (R)');
    ylabel('Mean Percent Error (MPE)');
    title('Generalization: MPE vs. Distance (R)');
    legend('show'); 
end