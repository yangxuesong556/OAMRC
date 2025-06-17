clc; clear; close all;

% Load data
data = load('../data/intrffic.txt');
data = data./max(data);

% Normalize the data to the range [-1, 1]
% Normalize train_data to [-1, 1]
% train_data = 2 * ((data - dataMin) / (dataMax - dataMin)) - 1;

% Extract high-frequency features
highFreqFeatures = extractHighFreqFeatures(data, 'db1');

% Define fixed parameters
trainLen = 8000; % Training length
testLen = 1000;  % Testing length
initLen = 10;   % Initial transient length
reg = 1e-8;      % Regularization coefficient
maxIterations = 10; % Maximum number of iterations for search
threshold  = 0.3;
% Define search space for parameters
resSizeRange = [100,500];
numLayersRange = [1, 3, 5];
aRange = [0.1, 0.3, 0.5, 0.7];
delayRange = [1,3,5];

% Initialize arrays to store parameter combinations and NRMSE
paramCombinations = [];
nrmseValues = [];

% Initialize best performance tracking
bestNRMSE = inf;
bestanomalies = [];
bestParams = struct('resSize', 0, 'numLayers', 0, 'a', 0, 'delay', 0);
bestY = []; % Store the best prediction result
Ypec = [];

% Start tracking total time for the random search process
totalStartTime = tic; % Start the overall timer

% Random search loop
for iter = 1:maxIterations
    % Start timer for each iteration
    iterStartTime = tic;

    % Randomly select parameters
    resSize = randsample(resSizeRange, 1);
    numLayers = randsample(numLayersRange, 1);
    a = randsample(aRange, 1);
    delay = randsample(delayRange, 1);

    % Run the model with the selected parameters
    [Y, Wout, mse, nrmse, flops, anomalies] = DAMRC_Anomaly(data, highFreqFeatures, trainLen, testLen, initLen, resSize, numLayers, a, reg, delay, threshold);

    % Perform denormalization of the prediction Y to restore the original scale
    % Y = ((Y + 1) / 2) * (dataMax - dataMin) + dataMin; % Denormalize the output prediction
    Ypec(iter,:) = Y';

    % Print current parameters and NRMSE for this iteration
    disp(['Iteration ', num2str(iter), ...
          ': resSize = ', num2str(resSize), ...
          ', numLayers = ', num2str(numLayers), ...
          ', a = ', num2str(a), ...
          ', delay = ', num2str(delay), ...
          ', NRMSE = ', num2str(nrmse)]);
    disp(['Iteration ', num2str(iter), ': FLOPs = ', num2str(flops)]);
    % Store parameter combination and NRMSE
    paramCombinations = [paramCombinations; resSize, numLayers, a, delay];
    nrmseValues = [nrmseValues; nrmse];

    % Check if this is the best result so far
    if nrmse < bestNRMSE
        bestNRMSE = nrmse;
        bestanomalies = anomalies;
        bestParams = struct('resSize', resSize, 'numLayers', numLayers, 'a', a, 'delay', delay);
        bestY = Y; % Store the prediction for the best parameters
    end

    % Stop timer for this iteration and display iteration time
    iterTime = toc(iterStartTime); % Time taken for this iteration
    disp(['Iteration time: ', num2str(iterTime), ' seconds']);
end

disp('Anomaly points (indices):');
% disp(anomalies);

% Plot results
% figure;
% plot(data, 'b');  % Plot true data
% hold on;
% plot(trainLen + (1:testLen), Y, 'r');  % Plot predicted data
% legend('True Data', 'Predicted Data');
figure; 
plot(data, 'b');  % Plot true data
hold on;
plot(trainLen + (1:testLen), bestY, 'k');  % Plot predicted data
legend('True Data', 'Predicted Data');

% Mark anomalies with different color and larger size
plot(trainLen + bestanomalies, bestY(bestanomalies), 'ro', 'MarkerSize', 10);  % Green circles for anomalies

% Optional: Add text annotations for anomalies
% for i = 1:length(bestanomalies)
%     text(trainLen + bestanomalies(i), Y(bestanomalies(i)), '  Anomaly', ...
%         'Color', 'green', 'FontSize', 12, 'FontWeight', 'bold');
% end
title('Anomaly Detection in Prediction');
% Function to extract high-frequency features using wavelet transform
function highFreqFeatures = extractHighFreqFeatures(data_train, waveletName)
    [C, L] = wavedec(data_train, 1, waveletName); % Decompose the signal
    highFreqFeatures = wrcoef('d', C, L, waveletName, 1); % Get high-frequency coefficients
end
