function [ ] = Generate_data_quadratic_memory_task(T)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
% Q=[0.2 0.1 0.1 0.1; 0.1 0.3 0.2 0.1; 0.1 0.2 0.4 0.1; 0.1 0.1 0.1 0.8];
% T=10000;
% u=unifrnd(0,0.5,[1,T+30]);
% y=[unifrnd(0,0.5,[1,30]),zeros(1,T)];
u=unifrnd(0,0.5,[1,T+30]);
y=[unifrnd(0,0.5,[1,30]),zeros(1,T)];
for i=30:T+29
    y(i+1)=0.2*y(i)+0.04*y(i)*sum(y(:,i-29:i))+1.5*u(i)*u(i-29)+0.001;
    %y(i+1)=tanh(0.3*y(i)+0.05*y(i)*sum(y(:,i-19:i))+1.5*u(i)*u(i-19)+0.01);
    %y(i+1)=0.3*y(i)+0.05*y(i)*sum(y(:,i-9:i))+1.5*u(i)*u(i-9)+0.1;

end
u=u(31:end);
y=y(31:end);
%plot(y)
save('D:\Matlab 2020b\R2020b\bin\TDR\DATDR\数据集\NARMA30data.mat','u','y')
end

