function [Y, Wout, mse, nrmse, flops, anomalies] = DAMRC_Anomaly(data, highFreqFeatures, trainLen, testLen, initLen, resSize, numLayers, a, reg, delay, threshold)
    % multiLayerOAMRC_with_anomaly_detection: A multi-layer OAMRC model with delay and state aggregation.
    % This function also detects anomalies based on prediction errors.
    
    % Normalize input data and high-frequency features
    dataMin = min(data);
    dataMax = max(data);
    data = (data - dataMin) / (dataMax - dataMin);  % Normalize data to [0, 1]
    
    highFreqFeaturesMin = min(highFreqFeatures(:));
    highFreqFeaturesMax = max(highFreqFeatures(:));
    highFreqFeatures = (highFreqFeatures - highFreqFeaturesMin) / (highFreqFeaturesMax - highFreqFeaturesMin);  % Normalize high-frequency features to [0, 1]

    % Dimensions
    inSize = 1;  % Input size (1 for the current input)
    m = size(highFreqFeatures, 2);  % Number of high-frequency features
    outSize = 1;  % Output size
    totalResSize = resSize * numLayers;  % Total reservoir size

    % Input weights
    Win = (randn(resSize, 1 + inSize + m) - 0.5) * 0.2; % Scale to [-0.1, 0.1]

    % State transition matrices for each layer
    W_layers = cell(numLayers, 1);
    baseSparsity = 0.1; % Base sparsity
    sparsityStep = 0.05; % Incremental sparsity per layer
    for l = 1:numLayers
        W = randn(resSize, resSize) * 0.1; % Random weights scaled to small values
        sparsity = min(baseSparsity + (l - 1) * sparsityStep, 1.0); % Increase sparsity per layer
        W(rand(size(W)) > sparsity) = 0; % Introduce sparsity
        spectral_radius = max(abs(eig(W)));
        W_layers{l} = W / spectral_radius * 0.9; % Scale to spectral radius < 1
    end

    % Initialize states
    X = zeros(1 + inSize + m + totalResSize, trainLen - initLen); % Include bias, input, features, and states
    Yt = data(initLen + 2:trainLen + 1)';  % Target outputs
    x_layers = zeros(resSize, numLayers);  % States for each layer
    flops_layer = resSize * resSize; % Each layer has O(resSize^2) FLOPs for matrix multiplication
    flops_total = 0; % Total FLOPs counter

    % Helper function for state updates
    function x_layers = update_states(u, hf, x_layers, Win, W_layers, a, delay)
        for l = 1:numLayers
            if l == 1
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(Win * [1; u; hf] + W_layers{l} * x_layers(:, l));
            else
                delayed_state = (1 - a)^delay * x_layers(:, l - 1) + a^delay * x_layers(:, l - 1);
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(W_layers{l} * delayed_state);
            end
            flops_total = flops_total + flops_layer;
        end
    end

    % Training: Collect states
    for t = 1:trainLen
        u = data(t);
        hf = highFreqFeatures(t, :)';  % Current high-frequency feature
        x_layers = update_states(u, hf, x_layers, Win, W_layers, a, delay);

        if t > initLen
            % Aggregate all layer states
            x_agg = reshape(x_layers, [], 1);  % Flatten state
            X(:, t - initLen) = [1; u; hf; x_agg];  % Include bias, input, features, and states
        end
    end

    % Train output weights using ridge regression
    X_T = X';
    Wout = Yt * X_T / (X * X_T + reg * eye(size(X, 1)));  % Regularized least squares
    % Testing: Generate predictions
    Y = zeros(outSize, testLen);
    u = data(trainLen + 1);
    for t = 1:testLen
        hf = highFreqFeatures(trainLen + t, :)';  % Current high-frequency feature
        x_layers = update_states(u, hf, x_layers, Win, W_layers, a, delay);
        x_agg = reshape(x_layers, [], 1);  % Flatten state
        Y(:, t) = Wout * [1; u; hf; x_agg];  % Predict output
        u = Y(:, t);  % Use predicted value as input
    end

    % Denormalize predictions
    Y = Y * (dataMax - dataMin) + dataMin;  % Reverse normalization to original scale
    data = data * (dataMax - dataMin) + dataMin;
    true_values = data(trainLen + 2:trainLen + testLen + 1)';
    mse = mean((true_values - Y).^2);
    rmse = sqrt(mse);
    nrmse = rmse / (max(true_values) - min(true_values));
    
    % Add final FLOPs count (includes output calculation)
    flops = flops_total + resSize * 1;
    
    % Anomaly Detection: Compare predictions with true values
    prediction_error = abs(true_values - Y);
    anomalies = find(prediction_error > threshold);  % Identify anomalies based on threshold
end
