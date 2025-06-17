function [ y] = Prediction_TDR(State0,Input_streaming,TPD,Noise,Input_Mask,W_out,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
trainLen=TPD(1);
predictLen=TPD(2);
resSize=round(Delay_Layer/h);
y=zeros(predictLen,1);

Temp=zeros(sum(resSize),1);
State1_n_1=State0(1:resSize,:);
for i=1:predictLen
    
    Temp=0*Temp;
    P1_Input_pattern=Input_streaming(trainLen+i);
    n=1;
    Input_pattern1=times(Input_Mask{n},P1_Input_pattern);

    [State1_n] = layer(Delay_Layer(n),State1_n_1,...
                Input_pattern1,Noise(1:resSize,trainLen+i),...
                gamma,alpha,h,p);
    Temp(1:resSize,1)=State1_n;
    State1_n_1=[State1_n_1(:,2),State1_n];
    y(i)=W_out'*[Temp;0];
end
    
end

