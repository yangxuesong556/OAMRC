clc; clear; close all;

% Load data
data = load('D:\yangxuesong\hlju_project\hlju_study\2025_Graduation_thesis\code\RC_code\data\EGC_data\ECG_yuan2.txt');
% Normalize the data to the range [-1, 1]
% Normalize train_data to [-1, 1]
% train_data = 2 * ((data - dataMin) / (dataMax - dataMin)) - 1;

% Extract high-frequency features
highFreqFeatures = extractHighFreqFeatures(data, 'db1');

% Define fixed parameters
trainLen = 10000; % Training length
testLen = 5000;  % Testing length
initLen = 100;   % Initial transient length
reg = 1e-5;      % Regularization coefficient
maxIterations = 20; % Maximum number of iterations for search

% Define search space for parameters
resSizeRange = [100,200,500];
numLayersRange = [5];
aRange = [0.1, 0.3, 0.5, 0.7];
delayRange = [1,3,5,10];

% Initialize arrays to store parameter combinations and NRMSE
paramCombinations = [];
nrmseValues = [];

% Initialize best performance tracking
bestNRMSE = inf;
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
    [Y, Wout, ~, nrmse,flops] = multiLayerOAMRC(data, highFreqFeatures, trainLen, testLen, initLen, resSize, numLayers, a, reg, delay);

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
        bestParams = struct('resSize', resSize, 'numLayers', numLayers, 'a', a, 'delay', delay);
        bestY = Y; % Store the prediction for the best parameters
    end

    % Stop timer for this iteration and display iteration time
    iterTime = toc(iterStartTime); % Time taken for this iteration
    disp(['Iteration time: ', num2str(iterTime), ' seconds']);
end

% Stop the overall timer and display total time
totalTime = toc(totalStartTime); % Total time for the random search process
disp(['Total time for random search: ', num2str(totalTime), ' seconds']);

% Display best parameters and NRMSE
disp(['Best NRMSE = ', num2str(bestNRMSE)]);
disp(['Best Parameters: resSize = ', num2str(bestParams.resSize), ...
      ', numLayers = ', num2str(bestParams.numLayers), ...
      ', a = ', num2str(bestParams.a), ...
      ', delay = ', num2str(bestParams.delay)]);

% Create the first subplot for true and predicted signals
figure(1); 
plot(data(trainLen + 2:trainLen + testLen + 1), 'color', [0, 0.75, 0]); % True signal in green
hold on;
plot(bestY, 'b'); % Predicted signal in blue
hold off;
title('Target and Predicted Signals', 'FontSize', 16, 'FontName', 'Times New Roman'); % Title font and size
legend('True Signal', 'Predicted Signal', 'FontSize', 14, 'FontName', 'Times New Roman'); % Legend
xlabel('Time', 'FontSize', 16, 'FontName', 'Times New Roman'); % X-axis label
ylabel('Amplitude', 'FontSize', 16, 'FontName', 'Times New Roman'); % Y-axis label

% Set Chinese font for the title and labels
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16); % Set axis font to Times New Roman for English text

% Calculate RMSE and NRMSE
true_values = data(trainLen + 2:trainLen + testLen + 1)';
mse = mean((true_values - bestY).^2);
rmse = sqrt(mse);
nrmse = rmse / (max(true_values) - min(true_values));
disp(['RMSE: ', num2str(rmse)]);
disp(['NRMSE: ', num2str(nrmse * 100), '%']); % NRMSE as percentage

% Generate the second subplot for heatmap or 3D surface plot of parameter effects
figure(2);
scatter3(paramCombinations(:,1), paramCombinations(:,2), paramCombinations(:,3), 50, nrmseValues, 'filled');
xlabel('resSize', 'FontSize', 16, 'FontName', 'Times New Roman'); % X-axis label
ylabel('numLayers', 'FontSize', 16, 'FontName', 'Times New Roman'); % Y-axis label
zlabel('a', 'FontSize', 16, 'FontName', 'Times New Roman'); % Z-axis label
title('3D Scatter of NRMSE across Parameters', 'FontSize', 16, 'FontName', 'Times New Roman'); % Title font and size
colorbar;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16); % Axis font

% Adjusting the figure size to fit SCI format
set(gcf, 'Position', [100, 100, 600, 400]); % Resize figure to proper dimensions (for SCI publication)


% Function to extract high-frequency features using wavelet transform
function highFreqFeatures = extractHighFreqFeatures(data_train, waveletName)
    [C, L] = wavedec(data_train, 1, waveletName); % Decompose the signal
    highFreqFeatures = wrcoef('d', C, L, waveletName, 1); % Get high-frequency coefficients
end
