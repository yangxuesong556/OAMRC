clc; clear; close all;

% Load data
data = load('D:\yangxuesong\hlju_project\hlju_study\2025_Graduation_thesis\code\RC_code\data\EGC_data\ECG_yuan.txt');

% Extract high-frequency features
highFreqFeatures = extractHighFreqFeatures(data, 'db1');

% Define fixed parameters
trainLen = 28000;
testLen = 2500;
initLen = 100;
reg = 1e-5;
maxIterations = 20;

% Define search space
resSizeRange = [100, 200, 500];
numLayersRange = [3];
aRange = [0.1,0.3];
delayRange = [1, 3, 5, 10];

% Initialize tracking variables
paramCombinations = [];
nrmseValues = [];
bestNRMSE = inf;
bestParams = struct();
bestY = [];

% Start search
totalStartTime = tic;
for iter = 1:maxIterations
    iterStartTime = tic;

    resSize = randsample(resSizeRange, 1);
    numLayers = randsample(numLayersRange, 1);
    a = randsample(aRange, 1);
    delay = randsample(delayRange, 1);

    [Y, Wout, ~, nrmse, flops] = multiLayerOAMRC(data, highFreqFeatures, trainLen, testLen, initLen, resSize, numLayers, a, reg, delay);

    disp(['Iteration ', num2str(iter), ': resSize = ', num2str(resSize), ...
          ', numLayers = ', num2str(numLayers), ...
          ', a = ', num2str(a), ', delay = ', num2str(delay), ...
          ', NRMSE = ', num2str(nrmse)]);

    paramCombinations = [paramCombinations; resSize, numLayers, a, delay];
    nrmseValues = [nrmseValues; nrmse];

    if nrmse < bestNRMSE
        bestNRMSE = nrmse;
        bestParams = struct('resSize', resSize, 'numLayers', numLayers, 'a', a, 'delay', delay);
        bestY = Y;
    end

    iterTime = toc(iterStartTime);
    disp(['Iteration time: ', num2str(iterTime), ' seconds']);
end
totalTime = toc(totalStartTime);
disp(['Total search time: ', num2str(totalTime), ' seconds']);
disp(['Best NRMSE = ', num2str(bestNRMSE)]);
disp(['Best Parameters: resSize = ', num2str(bestParams.resSize), ...
      ', numLayers = ', num2str(bestParams.numLayers), ...
      ', a = ', num2str(bestParams.a), ...
      ', delay = ', num2str(bestParams.delay)]);

% Extract ground truth
true_values = data(trainLen + 2:trainLen + testLen + 1)';

% Compute RMSE and NRMSE
mse = mean((true_values - bestY).^2);
rmse = sqrt(mse);
nrmse = rmse / (max(true_values) - min(true_values));
disp(['RMSE: ', num2str(rmse)]);
disp(['NRMSE: ', num2str(nrmse * 100), '%']);

% Compute absolute error
error = abs(true_values - bestY);

% --- Method 1: 2σ Rule (Gaussian-based) ---
threshold_sigma = mean(error) + 2 * std(error);
anomalies_sigma = find(error > threshold_sigma);

% --- Method 2: Median Absolute Deviation (MAD) ---
MAD_value = median(abs(error - median(error)));
threshold_mad = median(error) + 2 * MAD_value;
anomalies_mad = find(error > threshold_mad);

disp(['Anomalies detected (2σ method): ', num2str(length(anomalies_sigma)), ...
      ' / ', num2str(length(error)), ...
      ' (', num2str(length(anomalies_sigma) / length(error) * 100), '%)']);
disp(['Anomalies detected (MAD method): ', num2str(length(anomalies_mad)), ...
      ' / ', num2str(length(error)), ...
      ' (', num2str(length(anomalies_mad) / length(error) * 100), '%)']);

% --- Plot True vs. Predicted Signals with Anomalies ---
figure(1);
plot(true_values, 'color', [0, 0.75, 0]); hold on;
plot(bestY, 'b'); % Prediction
scatter(anomalies_mad, bestY(anomalies_mad), 50, 'r', 'filled'); % Anomalies
hold off;
title('True vs. Predicted Signals with Anomalies', 'FontSize', 16);
legend('True Signal', 'Predicted Signal', 'Anomalies (MAD method)', 'FontSize', 14);
xlabel('Time', 'FontSize', 16);
ylabel('Amplitude', 'FontSize', 16);

% --- Error Distribution ---
figure(2);
histogram(error, 50, 'FaceColor', [0.2 0.2 0.8]); hold on;
xline(threshold_sigma, '--r', '2σ Threshold', 'FontSize', 14);
xline(threshold_mad, '--g', 'MAD Threshold', 'FontSize', 14);
hold off;
title('Error Distribution with Anomaly Thresholds', 'FontSize', 16);
xlabel('Absolute Error', 'FontSize', 16);
ylabel('Frequency', 'FontSize', 16);
legend('Error Distribution', '2σ Threshold', 'MAD Threshold');

% --- Function to extract high-frequency features ---
function highFreqFeatures = extractHighFreqFeatures(data_train, waveletName)
    [C, L] = wavedec(data_train, 1, waveletName);
    highFreqFeatures = wrcoef('d', C, L, waveletName, 1);
end
