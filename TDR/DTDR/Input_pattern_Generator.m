function [  ] = Input_pattern_Generator(n,State1)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
global D_state;
D_state{n}= [D_state{n}(:,2:end),State1];
end
