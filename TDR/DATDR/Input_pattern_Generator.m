function [  ] = Input_pattern_Generator(n,State1)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
global D_state;
D_state{n-1}= [D_state{n-1}(:,2:end),State1];
end
