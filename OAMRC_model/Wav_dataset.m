clc; clear; close all; 
ECG200 = load('../data\EGC_data\mitbih_train.txt');
% Set parameters
Fs = 1000; % Sampling frequency
num_samples_train = 1000; % Number of training samples
num_samples_test = 500; % Number of testing samples
sample_length = 500; % Length of each sample
noise_level = 0.1; % Increased noise level

% Waveform parameters
frequencies = [10, 100, 500]; % Frequencies of waves
wave_types = {'sine1','sine2','sine3''square', 'triangle'}; % Waveform types
amplitude = 1;

% Initialize training and testing data
train_data = [];
test_data = [];

% Generate waveforms with fixed frequencies and types
for i = 1:3
    % Select frequency (10Hz, 50Hz, or 500Hz)
    freq = frequencies(i);
    
    % Create time vector
    t = (0:sample_length-1) / Fs;

    % Generate and store waveforms
    for wave_idx = 1:length(wave_types)
        wave_type = wave_types{wave_idx};
        
        % Generate waveform
        if strcmp(wave_type, 'sine1')
            wave = amplitude * sin(2 * pi * freq * t); % Sine wave
        elseif strcmp(wave_type, 'sine2')
            wave = amplitude * sin(2 * pi * freq * t); % Sine wave
        elseif strcmp(wave_type, 'sine3')
            wave = amplitude * sin(2 * pi * freq * t); % Sine wave
        elseif strcmp(wave_type, 'square')
            wave = amplitude * square(2 * pi * freq * t); % Square wave
        elseif strcmp(wave_type, 'triangle')
            wave = amplitude * sawtooth(2 * pi * freq * t, 0.5); % Triangle wave
        end
        
        % Add random phase shift
        phase_shift = rand * 2 * pi; 
        wave = wave + 0.1*sin(2 * pi * freq * t + phase_shift);

        % Apply random amplitude modulation
        % mod_amp = 1 + 0.5 * cos(2 * pi * (1:sample_length) / sample_length); % Modulation
        % wave = wave .* mod_amp; % Apply modulation

        % Add noise to the waveform
        noise = noise_level * randn(1, sample_length);
        wave_noisy = wave + noise;

        % Assign label based on frequency and waveform type
        label = (i - 1) * 3 + wave_idx - 1; % Sequential labels for each waveform type and frequency

        % Generate training and testing samples for each waveform
        for j = 1:num_samples_train
            sample = [wave_noisy, label]; % Combine noisy waveform data and label
            train_data = [train_data; sample]; % Add to training data
        end

        for j = 1:num_samples_test
            sample = [wave_noisy, label]; % Combine noisy waveform data and label
            test_data = [test_data; sample]; % Add to testing data
        end
    end
end

% Save data as txt files
save('train_data_fixed.txt', 'train_data', '-ascii');
save('test_data_fixed.txt', 'test_data', '-ascii');

disp('Training and testing data with fixed waveforms have been generated and saved.');

% Plot the noisy waveforms for visual inspection (showing a few samples)
figure('Position', [100, 100, 600, 800]); % Adjust the size of the figure

% Plot first training sample - 10Hz Sine Wave
subplot(5, 1, 1);
plot(train_data(1, :)); 
title('10Hz Sine Wave - Noisy', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 10);

% Plot second training sample - 50Hz Sine Wave
subplot(5, 1, 2);
plot(train_data(2, :));
title('50Hz Sine Wave - Noisy', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 10);

% Plot third training sample - 500Hz Sine Wave
subplot(5, 1, 3);
plot(train_data(3, :)); 
title('500Hz Sine Wave - Noisy', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 10);

% Plot fourth training sample - Square Wave (50Hz)
subplot(5, 1, 4);
plot(train_data(4, :)); 
title('50Hz Square Wave - Noisy', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 10);

% Plot fifth training sample - Triangle Wave (50Hz)
subplot(5, 1, 5);
plot(train_data(5, :)); 
title('50Hz Triangle Wave - Noisy', 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);

set(gcf, 'Color', 'w'); % Set figure background to white
% tight_layout(); % Adjust layout to avoid overlap
