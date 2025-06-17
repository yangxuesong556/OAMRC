clc
clear
epsilon=0;TPD=[3000,1000,200];
TDR_size=400;
 % training_data=readmatrix('F:\MATLAB\data\SN_ms_tot_V2.0.txt');
% data0=training_data(7:3248,4);
% training_data= readmatrix('F:\MATLAB\data\ETTh1.csv');
% data0=training_data(:,8);
% training_data= readmatrix('F:\MATLAB\data\ETT_de.txt');
% data0=training_data(: ,4);
% data0=load('F:\MATLAB\data\temperature.txt');
% % data0=load('F:\MATLAB\data\inttraffic.txt');
% data0=load('F:\MATLAB\data\MG17.txt');
% data0=load('F:\MATLAB\data\Internet.txt');
% data0=load('F:\MATLAB\data\temperature.txt');
% data0=load('F:\MATLAB\data\temperature.txt');
% data0=load('F:\MATLAB\data\inttraffic.txt');
data0=load('F:\MATLAB\data\MG17.txt');
% data0=load('F:\MATLAB\data\Internet.txt');
% data0=load('F:\MATLAB\data\temperature.txt');
% training_data=load('F:\MATLAB\data\lorzen_12.txt');
% data0=training_data(: ,3);
FB=84;
Q=[0.5,0.5];
input_Feedback=[1,1];
f_Feedback=0;
M=1000;
gamma=1.3;
alpha=1.5;
h=0.5;
p=1;
La=4;
De=1;
Delay_time=De*ones(1,La);
NUM=round(TDR_size/La*ones(1,La));
Delay_Layer=NUM*h;
% Deep_delay_double_TDR(epsilon,TPD,data,FB,gamma,alpha,h,p,Q,input_Feedback,f_Feedback,Delay_Layer,Delay_time,M)
for i=1:1
[err_nrmse(i)]=Deep_delay_double_TDR(epsilon,TPD,data0,FB,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time,M);
end
% mean_rmse=mean(err_rmse);
mean_nrmse=mean(err_nrmse);
 % disp(['the err_rmse: [' num2str(mean_rmse) ']'])
  disp(['the err_nrmse: [' num2str(mean_nrmse) ']'])