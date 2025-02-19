%assert(exist('DEBUG', 'var') == 1, 'you must run this script from src/main.m');
%assert(exist('dataset', 'var') == 1, 'dataset variable not found');
%disp('Asserts passed');
dataset = load("src/data/dataset.mat");


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


%Divide in train, validation and test
divs = surfaceNum / 3;
divr = randomNum / 3;

trainSTrj = surfaceTrj(1:divs,:);
trainSAcc = surfaceAcc(1:divs,:);
trainSPot = surfacePot(1:divs,:);
validationSTrj = surfaceTrj(divs+1:divs*2,:);
validationSAcc = surfaceAcc(divs+1:divs*2,:);
validationSPot = surfacePot(divs+1:divs*2,:);
testSTrj = surfaceTrj(divs*2+1:divs*3,:);
testSAcc = surfaceAcc(divs*2+1:divs*3,:);
testSPot = surfacePot(divs*2+1:divs*3,:);

trainRTrj = randomTrj(1:divr,:);
trainRAcc = randomAcc(1:divr,:);
trainRPot = randomPot(1:divr,:);
validationRTrj = randomTrj(divr+1:divr*2,:);
validationRAcc = randomAcc(divr+1:divr*2,:);
validationRPot = randomPot(divr+1:divr*2,:);
testRTrj = randomTrj(divr*2+1:divr*3,:);
testRAcc = randomAcc(divr*2+1:divr*3,:);
testRPot = randomPot(divr*2+1:divr*3,:);

%Concatenate matrices
trainTrj = cat(1,trainSTrj,trainRTrj);
trainAcc = cat(1,trainSAcc,trainRAcc);
trainPot = cat(1,trainSPot,trainRPot);
validationTrj = cat(1,validationSTrj,validationRTrj);
validationAcc = cat(1,validationSAcc,validationRAcc);
validationPot = cat(1,validationSPot,validationRPot);
testTrj = cat(1,testSTrj,testRTrj);
testAcc = cat(1,testSAcc,testRAcc);
testPot = cat(1,testSPot,testRPot);

%Min-Max scale
%minTrj = min(trainTrj);
%maxTrj = max(trainTrj);
%minAcc = min(trainAcc);
%maxAcc = max(trainAcc);
%minPot = min(trainPot);
%maxPot = max(trainPot);

%trainTrj = rescale(trainTrj,"InputMax",maxTrj,"InputMin",minTrj);
%validationTrj = rescale(validationTrj,"InputMax",maxTrj,"InputMin",minTrj);
%testTrj = rescale(testTrj,"InputMax",maxTrj,"InputMin",minTrj);
%trainAcc = rescale(trainAcc,"InputMax",maxAcc,"InputMin",minAcc);
%validationAcc = rescale(validationAcc,"InputMax",maxAcc,"InputMin",minAcc);
%testAcc = rescale(testAcc,"InputMax",maxAcc,"InputMin",minAcc);
%trainPot = rescale(trainPot,"InputMax",maxPot,"InputMin",minPot);
%validationPot = rescale(validationPot,"InputMax",maxPot,"InputMin",minPot);
%testPot = rescale(testPot,"InputMax",maxPot,"InputMin",minPot);


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

%Shuffle datastores
trainingSet = shuffle(trainingSet);
validationSet = shuffle(validationSet);
testSet = shuffle(testSet);