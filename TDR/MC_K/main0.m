
%Generate_data_quadratic_memory_task(5000)
clear
clc
 epsilon=0;
 TPD=[1500,110,300];
 FB=0;
NUM=[100];
M=2*FB+50;%200+50
Q=[0.5 0.5]; %比率
input_Feedback=[1,1];%双反馈环
f_Feedback=0;%反馈因子
Delay_time=[0];
h=0.3;
gamma=0.1;
alpha=0.3;
p=1;
for i=1:20
for k=1:100
Generate_data_quadratic_memory_task(5000,k)
training_data=load('D:\Matlab 2020b\R2020b\bin\TDR\DATDR\数据集\NARMA30data.mat');%MackyG17 Santa_Fe arfima
data=training_data;
Delay_Layer=NUM*h;
            [ nrmse_p(i,k),MCK1(i,k)]=Deep_delay_double_TDR(epsilon,TPD,data,FB,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time,M);
end
end

figure(1)
x1 = 1:k;
y1 = mean(MCK1(:,1:100));
sum(y1)%=9.0378
err = std(MCK1(:,1:100));
plot(x1,y1,'b','linewidth',0.5)
axis([0 100 0 1.1])

% clear
% for i=1:20;
% for k=1:50;
%     Generate_data_quadratic_memory_task(10000,k)
%     [nrmse_p1(i,k),MCK1(i,k)]=F_RCS(0.5,0.5,0.2,2); %100结点
% end
% end
