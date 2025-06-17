function [score,RMSE,NRMSE]=RC_main(SalpPositions,N)


NUM=1000;
n=1;
%% 参数
global sparsity;  %稀疏度
global spectral_radius;  % 最大谱半径
sparsity=SalpPositions(1);
spectral_radius=SalpPositions(2);
f_Feedb=SalpPositions(end);
input_dim=1;
output_dim=input_dim;
resSize=[200,200,200,200,200];
n_layers=length(resSize);
W_in=cell(1,n_layers);
W=cell(1,n_layers);
W_Feedb=cell(1,n_layers);
for i= 1:n_layers
    if i==1
        [W_in{i},W{i},W_Feedb{i}]=W_Generator(f_Feedb,input_dim,output_dim,resSize(i));
    else
        [W_in{i},W{i},W_Feedb{i}]=W_Generator(f_Feedb,resSize(i-1)+input_dim,output_dim,resSize(i));
    end
end
for i=400:25:500
    
    [RMSE(n),NRMSE(n)]=test_fun(i,NUM,SalpPositions,W_in,W,W_Feedb);
    n=n+1;
end
score=mean(RMSE);
end