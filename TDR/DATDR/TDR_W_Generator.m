function [W_in ] = TDR_W_Generator( input_dim,output_dim,hidden_dim )
%UNTITLED7 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% ��Ĥ
%global f_Feedback  %��������
W_in = (rand(hidden_dim, input_dim)  - 0.5)/5; %[-0.1, 0.1]
% �൱��win
% ��������
%W_Feedb = (rand(hidden_dim, output_dim) - 0.5); %[-0.5, 0.5].*f_Feedback

end

