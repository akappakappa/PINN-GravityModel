assert(1 == exist('dataset', 'var'), 'dataset variable not found');

% Variables
[tmp, train, test, validation, minmax] = deal(struct);

% Random indices
tmp.ssNum = size(dataset.surfaceTRJ, 1);
tmp.rrNum = size(dataset.randomdTRJ, 1);
tmp.ssIdx = randperm(tmp.ssNum);
tmp.rrIdx = randperm(tmp.rrNum);

% Shuffle according to random indices
tmp.ssTrj = dataset.surfaceTRJ(tmp.ssIdx, :);
tmp.ssAcc = dataset.surfaceACC(tmp.ssIdx, :);
tmp.ssPot = dataset.surfacePOT(tmp.ssIdx, :);
tmp.rrTrj = dataset.randomdTRJ(tmp.rrIdx, :);
tmp.rrAcc = dataset.randomdACC(tmp.rrIdx, :);
tmp.rrPot = dataset.randomdPOT(tmp.rrIdx, :);

% Split quantities percentages
tmp.qnt = [0.6, 0.2, 0.2];
assert(1 == sum(tmp.qnt), 'split percentages must sum to 1');
assert(0 == sum(tmp.qnt == 0), 'split percentages must be greater than 0');
train.ssDiv = floor(tmp.ssNum * tmp.qnt(1));
train.rrDiv = floor(tmp.rrNum * tmp.qnt(1));
validation.ssDiv = train.ssDiv + floor(tmp.ssNum * tmp.qnt(2));
validation.rrDiv = train.rrDiv + floor(tmp.rrNum * tmp.qnt(2));
% --> test division is the remaining data :end <--
% --> test division is the remaining data :end <--

% Split into training, validation and test sets
train.ssTrj      = tmp.ssTrj(1                   :train.ssDiv     , :);
train.ssAcc      = tmp.ssAcc(1                   :train.ssDiv     , :);
train.ssPot      = tmp.ssPot(1                   :train.ssDiv     , :);
train.rrTrj      = tmp.rrTrj(1                   :train.rrDiv     , :);
train.rrAcc      = tmp.rrAcc(1                   :train.rrDiv     , :);
train.rrPot      = tmp.rrPot(1                   :train.rrDiv     , :);
validation.ssTrj = tmp.ssTrj(train.ssDiv + 1     :validation.ssDiv, :);
validation.ssAcc = tmp.ssAcc(train.ssDiv + 1     :validation.ssDiv, :);
validation.ssPot = tmp.ssPot(train.ssDiv + 1     :validation.ssDiv, :);
validation.rrTrj = tmp.rrTrj(train.rrDiv + 1     :validation.rrDiv, :);
validation.rrAcc = tmp.rrAcc(train.rrDiv + 1     :validation.rrDiv, :);
validation.rrPot = tmp.rrPot(train.rrDiv + 1     :validation.rrDiv, :);
test.ssTrj       = tmp.ssTrj(validation.ssDiv + 1:end             , :);
test.ssAcc       = tmp.ssAcc(validation.ssDiv + 1:end             , :);
test.ssPot       = tmp.ssPot(validation.ssDiv + 1:end             , :);
test.rrTrj       = tmp.rrTrj(validation.rrDiv + 1:end             , :);
test.rrAcc       = tmp.rrAcc(validation.rrDiv + 1:end             , :);
test.rrPot       = tmp.rrPot(validation.rrDiv + 1:end             , :);

% Concatenate matrices
train.Trj      = cat(1, train.ssTrj, train.rrTrj);
train.Acc      = cat(1, train.ssAcc, train.rrAcc);
train.Pot      = cat(1, train.ssPot, train.rrPot);
validation.Trj = cat(1, validation.ssTrj, validation.rrTrj);
validation.Acc = cat(1, validation.ssAcc, validation.rrAcc);
validation.Pot = cat(1, validation.ssPot, validation.rrPot);
test.Trj       = cat(1, test.ssTrj, test.rrTrj);
test.Acc       = cat(1, test.ssAcc, test.rrAcc);
test.Pot       = cat(1, test.ssPot, test.rrPot);

% Min-Max scaling
[minmax.minTrj, minmax.maxTrj, minmax.minAcc, minmax.maxAcc, minmax.minPot, minmax.maxPot] ...
    = deal(min(train.Trj), max(train.Trj), min(min(train.Acc)), max(max(train.Acc)), min(train.Pot), max(train.Pot));
train.Trj      = rescale(train.Trj, "InputMax", minmax.maxTrj, "InputMin", minmax.minTrj);
train.Acc      = rescale(train.Acc, "InputMax", minmax.maxAcc, "InputMin", minmax.minAcc);
train.Pot      = rescale(train.Pot, "InputMax", minmax.maxPot, "InputMin", minmax.minPot);
validation.Trj = rescale(validation.Trj, "InputMax", minmax.maxTrj, "InputMin", minmax.minTrj);
validation.Acc = rescale(validation.Acc, "InputMax", minmax.maxAcc, "InputMin", minmax.minAcc);
validation.Pot = rescale(validation.Pot, "InputMax", minmax.maxPot, "InputMin", minmax.minPot);
test.Trj       = rescale(test.Trj, "InputMax", minmax.maxTrj, "InputMin", minmax.minTrj);
test.Acc       = rescale(test.Acc, "InputMax", minmax.maxAcc, "InputMin", minmax.minAcc);
test.Pot       = rescale(test.Pot, "InputMax", minmax.maxPot, "InputMin", minmax.minPot);

% Save to file
writematrix(train.Trj, "datastore/train/Trj.csv");
writematrix(train.Acc, "datastore/train/Acc.csv");
writematrix(train.Pot, "datastore/train/Pot.csv");
writematrix(validation.Trj, "datastore/validation/Trj.csv");
writematrix(validation.Acc, "datastore/validation/Acc.csv");
writematrix(validation.Pot, "datastore/validation/Pot.csv");
writematrix(test.Trj, "datastore/test/Trj.csv");
writematrix(test.Acc, "datastore/test/Acc.csv");
writematrix(test.Pot, "datastore/test/Pot.csv");

% Datastores
perm       = {tmp.ssIdx, tmp.rrIdx};
split      = [size(train.Trj, 1), size(validation.Trj, 1), size(test.Trj, 1)];
save("datastore/ds.mat", "perm", "split");

clear tmp train test validation minmax perm split;