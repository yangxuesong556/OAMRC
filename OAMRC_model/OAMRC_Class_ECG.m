% Inputs:
%   data: Time series input data
%   trainLen: Length of training data
%   testLen: Length of test data
%   initLen: Initial transient length
%   resSize: Reservoir size (neurons per layer)
%   numLayers: Number of layers in the ESN
%   a: Leaking rate
%   reg: Regularization coefficient
%   delay: Delay parameter between layers

clc;clear;close all;

%% ----------------------Data preparation-----------------------------
Test_Num = 1000; % Number of test samples
ECG200 = load('D:\yangxuesong\hlju_project\hlju_study\2025_Graduation_thesis\code\RC_code\data\EGC_data\mitbih_train.txt');
singlen = size(ECG200, 2); % Length of a single ECG record
Cut_Num = 165; % Length of the cut window
train_Num = 10000; % Number of training samples

% Hyperparameters for ESN
resSize = 100; % Reservoir size per layer
numLayers = 1; % Number of layers in the ESN
a = 0.1; % Leaking rate
reg = 1e-5; % Regularization coefficient
delay = 100; % Delay between layers

%% ----------------------Dataset Preparation-----------------------------
% Randomly shuffle the data
randIndex = randperm(size(ECG200, 1));
ECG200 = ECG200(randIndex, :);

% Prepare training data
trainECG = ECG200(1:train_Num, 1:Cut_Num); % ECG signal data for training
label_train = ECG200(1:train_Num, singlen); % Labels for training data

% One-Hot encoding of labels for multi-class classification
labels = [0, 1, 2, 3, 4]; % Label categories
one_hot_train = dummyvar(label_train + 1); % One-Hot encoding (MATLAB expects 1-based indexing)

% Load and shuffle test data
ECG200test = load('D:\yangxuesong\hlju_project\hlju_study\2025_Graduation_thesis\code\RC_code\data\EGC_data\mitbih_test.txt');
randIndex = randperm(size(ECG200test, 1));
ECG200test = ECG200test(randIndex, :);

% Prepare test data
testV1 = ECG200test(1:Test_Num, 1:Cut_Num); % ECG signal data for testing
label_test = ECG200test(1:Test_Num, singlen); % True labels for test data

% Define sizes
inSize = 1; % Input size (scalar input for each time step)
outSize = 1; % Output size
totalResSize = resSize * numLayers; % Total size of reservoir

%% ----------------------ESN Initialization-----------------------------
% Input weights for all layers
Win = (randn(resSize, 1 + inSize) + 0.5); % Random input weights

% Initialize state transition matrices for each layer
W_layers = cell(numLayers, 1); % Cell array to store matrices for each layer
for l = 1:numLayers
    W = eye(resSize) * 0.8; % Initialize W as an identity matrix scaled by 0.8
    Rand_W = randn(resSize, resSize); % Random state transition matrix
    W = Rand_W .* W; % Element-wise multiplication to adjust sparsity
    
    % Adding structured connections
    step = 5; % Connection step length
    for i = 1:step:resSize
        id = mod(i + step, resSize); % Ensure connections wrap around
        W(i, id) = 0.7; % Forward connection
        W(id, i) = 0.8; % Backward connection
    end
    W = W * 0.13; % Scale W to ensure stability
    W_layers{l} = W; % Store transition matrix for layer
end

%% ----------------------State Collection (Training)---------------------
% Initialize reservoir state storage
X = []; % Store state of the reservoir for each input
X_all = []; % Store aggregated states over all time steps

% State vectors for each layer
x_layers = zeros(resSize, numLayers);

% Run reservoir and collect states
for t = 1:size(trainECG, 1)
    for i = 1:size(trainECG, 2)
        u = trainECG(t, i); % Current input (ECG signal at time t)
        
        % Update states for each layer
        for l = 1:numLayers
            if l == 1
                % First layer receives direct input
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(Win * [1; u] + W_layers{l} * x_layers(:, l));
            else
                % Apply delay and state aggregation for subsequent layers
                delayed_state = x_layers(:, l - 1);
                for d = 1:min(delay, t)
                    delayed_state = (1 - a) * delayed_state + a * x_layers(:, l - 1); % Aggregated state with delay
                end
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(W_layers{l} * delayed_state); % Update current layer
            end
        end
        
        % Reshape state of all layers into a single vector
        x_proccess = reshape(x_layers, [], 1);
        X(:, i) = x_proccess; % Store state for current input
    end
    X_all(:, t) = reshape(X, [], 1); % Aggregate all states over time
end

% Visualize collected reservoir states
[time, lead] = meshgrid(1:size(X, 2), 1:size(X, 1));
figure;
surf(time, lead, X, 'EdgeColor', 'none'); % ??? surf ????3D?
view(60, 30); % ???????
colorbar; % ???????
xlabel('Lead'); % X ????
ylabel('Time'); % Y ????
zlabel('Amplitude'); % Z ????
title('3D Visualization of ECG Signal'); % ?????
axis tight; % ??????????
grid on; % ??????

figure;
imagesc(X); colorbar;
title('Reservoir States During Training', 'Times New Roman', 'FontSize', 14);
xlabel('Time Step','FontName', 'Times New Roman', 'FontSize', 14); 
ylabel('Neuron Index','FontName', 'Times New Roman', 'FontSize', 14);
figure;
plot(X')
title('Reservoir States During Training');
xlabel('Time Step'); ylabel('Neuron Index');
%% ----------------------Output Weight Training------------------------
% Ridge regression to compute output weights
tic;
I = eye(size(X_all, 1)); % Identity matrix for regularization
Wout = (one_hot_train' * X_all') / (X_all * X_all' + reg * I); % Solve using regularization
elapsedTime = toc;
disp(['??????????:',num2str(elapsedTime),'??'])

%% ----------------------State Collection (Testing)---------------------
% Initialize state storage for test data
X_test = [];
X_all_test = [];

for t = 1:size(testV1, 1)
    for i = 1:size(testV1, 2)
        u = testV1(t, i); % Current input (test signal at time t)
        
        % Update states for each layer
        for l = 1:numLayers
            if l == 1
                % First layer receives direct input
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(Win * [1; u] + W_layers{l} * x_layers(:, l));
            else
                % Apply delay and state aggregation for subsequent layers
                delayed_state = x_layers(:, l - 1);
                for d = 1:min(delay, t)
                    delayed_state = (1 - a) * delayed_state + a * x_layers(:, l - 1); % Aggregated state with delay
                end
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(W_layers{l} * delayed_state); % Update current layer
            end
        end
        
        % Reshape state of all layers into a single vector
        x_proccess = reshape(x_layers, [], 1);
        X_test(:, i) = x_proccess; % Store state for current input
    end
    X_all_test(:, t) = reshape(X_test, [], 1); % Aggregate all states over time
end

% Test predictions using the trained output weights
test_result_real = Wout * X_all_test; % Predict continuous output
[~, test_result] = max(test_result_real, [], 1); % Get the class with the highest score
test_result = test_result' - 1; % Convert to original label space

% Calculate accuracy
accuracies = sum(test_result == label_test) / Test_Num;
disp(['Final accuracy = ', num2str(accuracies)]);

%% ----------------------3D Scatter Plot Visualization------------------------
% Random 3D scatter plot to visualize results
x = rand(Test_Num, 1);
y = rand(Test_Num, 1);
z = test_result; % Predicted labels

figure;
hold on;
colors = ['r', 'g', 'b', 'm', 'c']; % Colors for different labels
markers = ['o', 's', 'd', '^', 'v']; % Markers for different labels
for i = 1:length(labels)
    idx = (label_test == labels(i));
    scatter3(x(idx), y(idx), z(idx), 50, colors(i), markers(i), 'filled');
end
xlabel('Random X projection');
ylabel('Random Y projection');
zlabel('Predicted values');
title('3D Scatter Plot of Predicted vs True Labels');
legend({'0', '1', '2', '3', '4'}, 'Location', 'best');
grid on;
hold off;

%% ----------------------2D Bubble Plot Visualization------------------------
% Random 2D bubble plot to visualize clustering of real and predicted labels

% Generate random 2D coordinates but keep the same label clustered together
x = zeros(Test_Num, 1);
y = zeros(Test_Num, 1);

% Define cluster centers for each label to create visual separation
cluster_centers = [0.2, 0.2; 0.5, 0.5; 0.8, 0.2; 0.2, 0.8; 0.8, 0.8];  % Centers for 5 categories

% Assign random points around each cluster center based on the label
for i = 1:length(labels)
    idx = (label_test == labels(i));  % Get indices of the true labels
    num_points = sum(idx);  % Number of points for the current label
    x(idx) = cluster_centers(i, 1) + 0.05 * randn(num_points, 1);  % Random offset for x
    y(idx) = cluster_centers(i, 2) + 0.05 * randn(num_points, 1);  % Random offset for y
end

% Set bubble sizes and transparency for better visualization
bubble_sizes = 100 * ones(Test_Num, 1);  % Increased size for better visibility
transparency = 0.6;  % Set transparency level

% Create the bubble plot
figure;
hold on;
colors = ['r', 'g', 'b', 'm', 'c'];  % Colors for different labels
markers_real = ['o', 'o', 'o', 'o', 'o'];  % Marker for real labels


% Plot the real labels with transparency
for i = 1:length(labels)
    idx = (label_test == labels(i));  % Get indices of each real label
    scatter(x(idx), y(idx), bubble_sizes(idx), colors(i), markers_real(i), ...
        'filled', 'MarkerFaceAlpha', transparency, 'MarkerEdgeAlpha', transparency, ...
        'DisplayName', ['True ', num2str(labels(i))]);  % Plot real labels with transparency
end
% Plot formatting
xlabel('X-Axis Projection', 'FontName', 'Times New Roman', 'FontSize', 16);
ylabel('Y-Axis Projection', 'FontName', 'Times New Roman', 'FontSize', 16);
title('2D Bubble Plot: True vs Predicted Labels', 'FontName', 'Times New Roman', 'FontSize', 18);
legend('Location', 'best', 'FontSize', 12);
grid on;
hold off;
figure;
hold on;
colors = ['r', 'g', 'b', 'm', 'c'];  % Colors for different labels
markers_pred = ['x', 'x', 'x', 'x', 'x'];  % Marker for predicted labels
% Overlay the predicted labels using a different marker ('x') and increased transparency
for i = 1:length(labels)
    idx = (test_result == labels(i));  % Get indices of each predicted label
    scatter(x(idx), y(idx), bubble_sizes(idx), colors(i), markers_pred(i), ...
        'MarkerFaceAlpha', transparency, 'MarkerEdgeAlpha', transparency, ...
        'DisplayName', ['Predicted ', num2str(labels(i))]);  % Plot predicted labels
end

% Plot formatting
xlabel('X-Axis Projection', 'FontName', 'Times New Roman', 'FontSize', 16);
ylabel('Y-Axis Projection', 'FontName', 'Times New Roman', 'FontSize', 16);
title('2D Bubble Plot: True vs Predicted Labels', 'FontName', 'Times New Roman', 'FontSize', 18);
legend('Location', 'best', 'FontSize', 12);
grid on;
hold off;

figure;
cm = confusionchart(test_result, label_test);
cm.Title = 'Final Combined Confusion Matrix';
cm.ColumnSummary = "column-normalized";
% cm.RowSummary = "row-normalized";
% ----------------------recall---------------------------
recall = zeros(length(labels), 1);
for i = 1:length(labels)
    %True Positive (TP) ?? False Negative (FN)
    TP = sum((test_result == labels(i)) & (label_test == labels(i))); % True Positive
    FN = sum((test_result ~= labels(i)) & (label_test == labels(i))); % False Negative
    
    % recall
    recall(i) = TP / (TP + FN);
    fprintf('Recall for label %d: %.2f%%\n', labels(i), recall(i) * 100);
end

% mean recall
mean_recall = mean(recall);
fprintf('Mean Recall: %.2f%%\n', mean_recall * 100);

%% ----------------------Confusion Matrix and Recall-------------------
% Plot confusion matrix
cm = confusionchart(test_result, label_test);
cm.Title = 'Final Combined Confusion Matrix';
cm.ColumnSummary = "column-normalized";

% Calculate recall for each label
recall = zeros(length(labels), 1);
for i = 1:length(labels)
    TP = sum((test_result == labels(i)) & (label_test == labels(i))); % True Positive
    FN = sum((test_result ~= labels(i)) & (label_test == labels(i))); % False Negative
    recall(i) = TP / (TP + FN);
    fprintf('Recall for label %d: %.2f%%\n', labels(i), recall(i) * 100);
end
mean_recall = mean(recall);
disp(['Mean recall: ', num2str(mean_recall * 100), '%']);

%% ----------------------Results Visualization-------------------------
% Visualize results in a 2D scatter plot
figure;
scatter(1:Test_Num, label_test, 'g', 'DisplayName', 'True Labels');
hold on;
scatter(1:Test_Num, test_result, 'r', 'DisplayName', 'Predicted Labels');
legend;
title('True vs Predicted Labels');
xlabel('Sample Index');
ylabel('Class Label');