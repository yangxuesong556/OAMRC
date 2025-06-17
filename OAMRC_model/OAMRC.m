function [Y, Wout, mse, nrmse] = multiLayerOAMRC(data, highFreqFeatures, trainLen, testLen, initLen, resSize, numLayers, a, reg, delay)
    % multiLayerOAMRC: A multi-layer OAMRC model with delay and state aggregation.
    %
    % Inputs:
    %   data: Time series input data
    %   highFreqFeatures: High-frequency features to enhance predictions
    %   trainLen: Length of training data
    %   testLen: Length of test data
    %   initLen: Initial transient length
    %   resSize: Reservoir size (neurons per layer)
    %   numLayers: Number of layers in the OAMRC
    %   a: Leaking rate
    %   reg: Regularization coefficient
    %   delay: Delay parameter between layers
    %
    % Outputs:
    %   Y: Predicted output
    %   Wout: Output weight matrix
    %   mse: Mean Squared Error
    %   nrmse: Normalized Root Mean Square Error

    inSize = 1; % Input size (1 for the current input)
    m = size(highFreqFeatures, 2);  % Number of high-frequency features
    outSize = 1; % Output size
    totalResSize = resSize * numLayers; % Total reservoir size

    % Input weights for all layers
    Win = (randn(resSize, 1 + inSize + m) + 0.5); % Include high-frequency features in Win
    % State transition matrices for each layer
    W_layers = cell(numLayers, 1);
    for l = 1:numLayers
        W = eye(resSize) * 0.8;
        Rand_W = randn(resSize, resSize);
        W = Rand_W .* W;

        step = 5;
        for i = 1:step:resSize
            id = mod(i + step, resSize);
            if ~isnumeric(id) || id <= 0 || id > size(W, 2)
                error(['Invalid index id: ', num2str(id), '. It should be a positive integer within W matrix dimensions.']);
            end

            W(i, id) = 0.7;
            W(id, i) = 0.5;
        end
        W = W * 0.13;  % Scale W
        W_layers{l} = W;
    end

    % Initialize states for all layers
    X = zeros(1 + inSize + m + totalResSize, trainLen - initLen); % Adjusted for features
    Yt = data(initLen + 2:trainLen + 1)';  % Target outputs
    x_layers = zeros(resSize, numLayers);  % States for each layer

    % Run reservoir and collect states
    for t = 1:trainLen
        u = data(t);
        hf = highFreqFeatures(t, :)';  % Current high-frequency feature

        % Update states for each layer
        for l = 1:numLayers
            if l == 1
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(Win * [1; u; hf] + W_layers{l} * x_layers(:, l));
            else
                delayed_state = x_layers(:, l - 1);
                for d = 1:delay
                    delayed_state = (1 - a) * delayed_state + a * x_layers(:, l - 1);
                end
                % State aggregation: combine previous and current layer
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(W_layers{l} * delayed_state);
            end
        end

        if t > initLen
            % Aggregate all layer states
            x_agg = reshape(x_layers, [], 1);  % Flatten state
            concatenated_vector = [1; u; hf; x_agg];  % Include bias, input, features, and states
            if length(concatenated_vector) ~= size(X, 1)
                error('Dimension mismatch: Concatenated vector size = %d, Expected size = %d', length(concatenated_vector), size(X, 1));
            end
            X(:, t - initLen) = concatenated_vector;  % Assign to X
        end
    end

    % Train output with ridge regression
    X_T = X';
    Wout = Yt * X_T / (X * X_T + reg * eye(1 + inSize + m + totalResSize));  % Updated with features

    % Test phase
    Y = zeros(outSize, testLen);
    u = data(trainLen + 1);
    hf = highFreqFeatures(trainLen + 1, :)';  % Initialize first high-frequency feature for testing

    for t = 1:testLen
        for l = 1:numLayers
            if l == 1
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(Win * [1; u; hf] + W_layers{l} * x_layers(:, l));
            else
                delayed_state = x_layers(:, l - 1);
                for d = 1:delay
                    delayed_state = (1 - a) * delayed_state + a * x_layers(:, l - 1);
                end
                x_layers(:, l) = (1 - a) * x_layers(:, l) + a * tanh(W_layers{l} * delayed_state);
            end
        end
        x_agg = reshape(x_layers, [], 1);  % Flatten state
        y = Wout * [1; u; hf; x_agg];  % Include bias, input, features, and states
        Y(:, t) = y;

        % Generative mode: use prediction as next input
        u = y;  % Update input with the prediction
        hf = highFreqFeatures(trainLen + t + 1, :)';  % Get next high-frequency feature
    end

    % MSE calculation
    errorLen = min(testLen, length(data) - trainLen - 1);
    mse = sum((data(trainLen + 2:trainLen + errorLen + 1)' - Y(1, 1:errorLen)).^2) / errorLen;

    % NRMSE calculation
    true_values = data(trainLen + 2:trainLen + errorLen + 1)';
    rmse = sqrt(mean((true_values - Y(1, 1:errorLen)).^2));
    nrmse = rmse / (max(true_values) - min(true_values));

    % Plot results: Predicted vs True
    % figure;
    % plot(data(trainLen + 2:trainLen + testLen + 1), 'color', [0, 0.75, 0]);
    % hold on;
    % plot(Y', 'r');
    % hold off;
    % title('Target and Predicted Signals');
    % legend('True Signal', 'Predicted Signal');
    % xlabel('Time');
    % ylabel('Amplitude');

    % % Plot output weights
    % figure;
    % plot(Wout');
    % title('Output Weights W^{out}');
    % xlabel('Neuron Index');
    % ylabel('Weight Value');
end
