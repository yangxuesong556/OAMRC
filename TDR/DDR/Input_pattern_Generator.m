function [  ] = Input_pattern_Generator(n,State1)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
global D_state;
D_state{n}= [D_state{n}(:,2:end),State1];
end
