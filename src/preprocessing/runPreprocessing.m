assert(exist('DEBUG', 'var') == 1, 'you must run this script from src/main.m');
assert(exist('dataset', 'var') == 1, 'dataset variable not found');
disp('Asserts passed');

%Get shuffle indices
surfaceNum = 200700;
randomNum = 90000;
idxSurf = randperm(surfaceNum);
idxRand = randperm(randomNum);

%Shuffle values
surfaceTrj = dataset.surfaceTRJ(idxSurf,:);
surfaceAcc = dataset.surfaceACC(idxSurf,:);
surfacePot = dataset.surfacePOT(idxSurf,:);
randomTrj = dataset.randomdTRJ(idxRand,:);
randomAcc = dataset.randomdACC(idxRand,:);
randomPot = dataset.randomdPOT(idxRand,:);

%Concatenate matrices
totalTrj = cat(1,surfaceTrj,randomTrj);
totalAcc = cat(1,surfaceAcc,randomAcc);
totalPot = cat(1,surfacePot,randomPot);

%Divide in train, validation and test
div = length(totalTrj) / 3;
trainTrj = totalTrj(1:div,:);
trainAcc = totalAcc(1:div,:);
trainPot = totalPot(1:div,:);
validationTrj = totalTrj(div+1:div*2,:);
validationAcc = totalAcc(div+1:div*2,:);
validationPot = totalPot(div+1:div*2,:);
testTrj = totalTrj(div*2+1:div*3,:);
testAcc = totalAcc(div*2+1:div*3,:);
testPot = totalPot(div*2+1:div*3,:);

%Initialize datastores
trainTrj = arrayDatastore(trainTrj);
trainAcc = arrayDatastore(trainAcc);
trainPot = arrayDatastore(trainPot);
validationTrj = arrayDatastore(validationTrj);
validationAcc = arrayDatastore(validationAcc);
validationPot = arrayDatastore(validationPot);
testTrj = arrayDatastore(testTrj);
testAcc = arrayDatastore(testAcc);
testPot = arrayDatastore(testPot);
trainingSet = combine(trainTrj,trainAcc,trainPot);
validationSet = combine(validationTrj,validationAcc,validationPot);
testSet = combine(testTrj,testAcc,testPot);
