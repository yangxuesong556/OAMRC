clear;close all;clc;

% load the data
trainLen = 3000;
testLen = 500;
initLen = 200;
% data = data.y;
% data = data';
data = load('../data/MackeyGlass_t17.txt');
% load('signv.mat');
% plot some of it
figure(1);
plot(data(1:1000));
title('A sample of data');

% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 100;
a = 0.3; % leaking rate

signal = data(1:500); %signal of W_in
rand( 'seed', 42 );
% Win = (zeros(resSize,1+inSize)+0.5) ; %r=0.5
Win = (randn(resSize,1+inSize)+0.5) ; %r=0.5
% W = eye(resSize) * 0.8;
W = diag(ones(1, resSize-1), 1); 
Rand_W = randn(resSize,resSize);
W = Rand_W.*W;

step = 5;
for i=1:step:resSize
    id = mod(i+step, resSize);
    W(i,id) = 0.7;
    W(id,i) = 0.5;
end

% Option 1 - direct scaling (quick&dirty, reservoir-specific):
W = W .* 0.13;
% Option 2 - normalizing and setting spectral radius (correct, slower):
disp 'Computing spectral radius...';
%opt.disp = 0;
%hoW = abs(eigs(W,1,'LM',opt));
% disp 'done.'
%W = W .* ( 1.25 /rhoW);
%W(500,1) = 0;
% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
    u = data(t);
    x = (1-a)*x + a*tanh( Win*[1;u] + (W*x) );
    if t > initLen
        X(:,t-initLen) = [1;u;x];
    end
end

% train the output
reg = 1e-8;  % regularization coefficient
X_T = X';
% Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
Wout = Yt*X_T / (X*X_T + reg*eye(1+inSize+resSize));
% Wout = Yt*pinv(X);

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = data(trainLen+1);
for t = 1:testLen 
    x = (1-a)*x + a*tanh( Win*[1;u] + W*x );
    y = Wout*[1;u;x];
    Y(:,t) = y;
    % generative mode:
    u = y;
    % this would be a predictive mode:
    %u = data(trainLen+t+1);
end

errorLen = 500;
mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );

% plot some signals
figure(1);
plot( data(trainLen+2:trainLen+testLen+1), 'color', [0,0.75,0] );
hold on;
plot( Y', 'r' );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');

figure(2);
plot( X(2:8,:)' );
title('Some reservoir activations x(n)');

figure(3);
% bar( Wout' )
plot( Wout' )
title('Output weights W^{out}');