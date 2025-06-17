function [nrmse]=Deep_delay_double_TDR(epsilon,TPD,data,FB,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time,M)

Training_steps=TPD(1);%;训练数
Predicting_steps=TPD(2);%测试数
T=Training_steps+Predicting_steps;%训练集加测试集
discarded_steps=TPD(3); %丢弃数
Training_data0=data(FB+1:T+FB+1)';
data0=(data-min(data))./(max(data)-min(data)); 
Input_streaming=data0(1:T+1)';
Training_data=data0(FB+1:T+FB+1)';
%% narma10 test
%     Input_streaming=data.u(:,1:T)'; %输入
%     Training_data=data.y(:,1:TPD(1))'; %输出数据

%% 生成连接矩阵 
resSize=round(Delay_Layer/h);
n_layers=length(Delay_Layer);
Input_Mask=cell(1,n_layers);
input_dim=1;output_dim=1;
for i= 1:n_layers 
     [Input_Mask{i}]=TDR_W_Generator(input_dim,output_dim,resSize(i));
%       Mask=load('C:\Users\17519\Desktop\分数阶MG方程_储备池\基于MG多反馈换的NARMA30测试\MAT_MG_RC\Generating_Data\MASK.mat');
%     if i==1
%         Input_Mask{i}=Mask.m(1:resSize(i));
%     else
%         Input_Mask{i}=Mask.m(sum(resSize(1:i-1))+1:sum(resSize(1:i)));
%     end
end
%% 噪声生成
Noise=0+sqrt(epsilon).*randn(sum(resSize),T);  %噪声
%% 训练阶段
[State]=train_TDR(TPD,Input_streaming,Noise,Input_Mask,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time);


%%  训练权重------------------------------------------------------------------------------------------------------------

input_dim=1;
XT=State(:,discarded_steps+1:end);YT=Training_data(discarded_steps+1:TPD(1));
reg = 1e-8;
W_out = ((XT*XT' + reg*eye(sum(resSize)+input_dim)) \ (XT*YT'));
% State=State(1:end-1,:);
% XT=State(:,discarded_steps+1:end);YT=Training_data(discarded_steps+1:TPD(1));
% reg = 1e-8;
% W_out = ((XT*XT' + reg*eye(sum(resSize))) \ (XT*YT));


%% 预测阶段
State0=State(:,end);
[ y] = Prediction_TDR(State0,Input_streaming,TPD,Noise,Input_Mask,W_out,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time);

y_trained0=y.*(max(Training_data0)-min(Training_data0))+min(Training_data0);
% y_trained0=y;
Refernce_p=Training_data0(TPD(1)+1:T);
% Refernce_p=data.y(:,TPD(1)+1:sum(TPD(1:2)));%参考
Test_p=y_trained0';%测试
nrmse_p=sqrt(mean((Test_p(1:M)-Refernce_p(1:M)).^2));
nrmse=sqrt(mean((Test_p(1:M)-Refernce_p(1:M)).^2)/mean((Refernce_p(1:M)-mean(Refernce_p(1:M))).^2));
% nrmse=sqrt(mean((Test_t(1:M)-Refernce_t(1:M)).^2)/mean((Refernce_t(1:M)-mean(Refernce_t(1:M))).^2));
%fprintf('N0=%d FB=%d De=%d M=%d  h=%f  g=%f  a=%f p=%d  F_nrmse_p=%f \n',input_Feedback(2),FB,Delay_time(1),M,h,gamma,alpha,p,nrmse_p)
% plot(Refernce_p(1:M),'b')
% hold on
% plot(y_trained0(1:M),'r')
end
