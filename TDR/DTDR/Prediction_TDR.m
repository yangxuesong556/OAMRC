function [ y] = Prediction_TDR(State0,Input_streaming,TPD,Noise,Input_Mask,W_out,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
global D_state;
n_Layer=length(Delay_Layer); %储备池层数
trainLen=TPD(1);
predictLen=TPD(2);
resSize=round(Delay_Layer/h);
y=zeros(predictLen,1);
for i=1:predictLen
    Temp=[];
    for n=1:n_Layer
        if n==1
            Input=Input_streaming(i+trainLen);
            noise=Noise(1:resSize(n),i+trainLen);
        else
            Input=State1_n(1);
            noise=Noise(sum(resSize(1:n-1))+1:sum(resSize(1:n)),i+trainLen);
        end
        Input_pattern1=times(Input_Mask{n},Input);
        [State1_n] = layer(Delay_Layer(n),D_state{n}(:,1:2),...
                Input_pattern1,noise,...
                gamma,alpha,h,p);
        Input_pattern_Generator( n,State1_n);
        Temp=[Temp;State1_n];
    end
    y(i)=W_out'*[Temp;0];
end
    
end

