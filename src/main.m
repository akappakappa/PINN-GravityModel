% --- Diff eq.
% J(Θ) = 1/Nf*[∑(i=1,Nf)|ai+∇Û(ri|Θ)|^2]

% --- Network
% N {10,20,40,80} per layer, 8 hidden layers

% --- Hyperparameters
% BatchSize = 262144 -> for little available VRAM
% LearningRate η0 = 0.005 -> decay after epoch i>=i0 : ηi = η0*pow(α,-(i-i0)/σ)
% ReferenceEpoch i0 = 25000, ScaleFactor σ = 25000, DecayRate α = 0.5
% ActivationFunction = GELU
% Epochs = 100000
% Optimizer = Adam
% Initializer = Glorot uniform
% x Transform (preprocessing to fit r) = MinMax along each component
% y Transform (preprocessing to fit a) = uniform MinMax across all components