function [NRMSE]=TDR_main(pop)

% training_data=readmatrix('F:\MATLAB\data\SN_ms_tot_V2.0.txt');
% data0=training_data(7:3248,4);
% training_data= readmatrix('F:\MATLAB\data\ETTh1.csv');
% data0=training_data(:,8);
% training_data= readmatrix('F:\MATLAB\data\ETT_de.txt');
% data0=training_data(: ,4);
% data0=load('F:\MATLAB\data\temperature.txt');
% data0=load('F:\MATLAB\data\inttraffic.txt');
data0=load('F:\MATLAB\data\MG17.txt');
% data0=load('F:\MATLAB\data\Internet.txt');
% data0=load('F:\MATLAB\data\temperature.txt');
% data0=load('F:\MATLAB\data\temperature.txt');
% data0=load('F:\MATLAB\data\inttraffic.txt');
% data0=load('F:\MATLAB\data\MG17.txt');
% data0=load('F:\MATLAB\data\Internet.txt');
% data0=load('F:\MATLAB\data\temperature.txt');
 TDR_size=400;
 epsilon=0;
 TPD=[3000,1000,200];
 FB=84;
M=1000;%200+50
%Q=[0.5 0.5]; %比率
%input_Feedback=[1,1];%双反馈环
%f_Feedback=0;%反馈因子
[n]=size(pop,1);
for i=1:n
    h=pop(i,1);
    gamma=pop(i,2);
    alpha=pop(i,3);
    p=pop(i,4);
    Delay_time=pop(i,5)*ones(1,pop(i,6));
    NUM=round(TDR_size/pop(i,6)*ones(1,pop(i,6)));
    Delay_Layer=NUM*h;
    [NRMSE(i)]=Deep_delay_double_TDR(epsilon,TPD,data0,FB,...
    gamma,alpha,h,p,Delay_Layer,Delay_time,M);
end

end