function [NRMSE]=TDR_main(pop)

training_data=load('C:\Users\17519\Desktop\第三篇代码\narma10-30修\TDR\DATDR\数据集\NARMA30data.mat');%MackyG17 Santa_Fe arfima
 data=training_data;

 TDR_size=200;
 epsilon=0;
 TPD=[3500,1000,500];
 FB=0;
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
    [NRMSE(i)]=Deep_delay_double_TDR(epsilon,TPD,data,FB,...
    gamma,alpha,h,p,Delay_Layer,Delay_time,M);
end

end