assert(1 == exist('dataset', 'var'), 'dataset variable not found');

% TODO: Load dataset

% Options
splitPercentages = [0.6, 0.2, 0.2];
assert(1 == sum(splitPercentages), 'split percentages must sum to 1');
assert(0 == sum(0 == splitPercentages), 'split percentages must be greater than 0');

% Call preprocessing function preset
[preprocessed, split] = presets.PINN_GM_III(dataset, splitPercentages);

% Save preprocessed dataset
writematrix(preprocessed.trainTRJ     , "datastore/train/Trj.csv"     );
writematrix(preprocessed.trainACC     , "datastore/train/Acc.csv"     );
writematrix(preprocessed.trainPOT     , "datastore/train/Pot.csv"     );
writematrix(preprocessed.validationTRJ, "datastore/validation/Trj.csv");
writematrix(preprocessed.validationACC, "datastore/validation/Acc.csv");
writematrix(preprocessed.validationPOT, "datastore/validation/Pot.csv");
writematrix(preprocessed.testTRJ      , "datastore/test/Trj.csv"      );
writematrix(preprocessed.testACC      , "datastore/test/Acc.csv"      );
writematrix(preprocessed.testPOT      , "datastore/test/Pot.csv"      );
save("datastore/ds.mat", "split");

clear;