
function [W_in, W, W_Feedb] = W_Generator(f,Input_dim,Output_dim,hidden_dim)
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
% 内部连接矩阵
global sparsity %稀疏度
global spectral_radius %谱半径
%% 内部连接矩阵 Internal connection matrix
W=rand(hidden_dim,hidden_dim)-0.5;
W(rand(hidden_dim,hidden_dim)<sparsity)=0;
radius=max(abs(eig(W)));
W = W * (spectral_radius / radius);
%% 掩膜 输入权重 Input Mask
W_in = (rand(hidden_dim, Input_dim)  - 0.5)/5;
%% 反馈矩阵 Feedback connection matrix.
% f is the factor that contral the  feedback intensity.
W_Feedb = f*(rand(hidden_dim, Output_dim) * 2 - 1);
end
%这个模块主要是写了W的生成无论是W_in ,还是储备池内部的连接矩阵W，还是最后的反馈矩阵W_Feedb，这里都给于说明
