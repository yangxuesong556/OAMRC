function [W_in ] = TDR_W_Generator( input_dim,output_dim,hidden_dim )
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
% 掩膜
%global f_Feedback  %反馈因子
W_in = (rand(hidden_dim, input_dim)  - 0.5)/5; %[-0.1, 0.1]
% 相当于win
% 反馈矩阵
%W_Feedb = (rand(hidden_dim, output_dim) - 0.5); %[-0.5, 0.5].*f_Feedback

end

