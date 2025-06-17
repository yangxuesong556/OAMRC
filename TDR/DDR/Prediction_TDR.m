function [ y] = Prediction_TDR(State0,Input_streaming,TPD,Noise,Input_Mask,W_out,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
global D_state;
n_Layer=length(Delay_Layer); %�����ز���
trainLen=TPD(1);
predictLen=TPD(2);
resSize=round(Delay_Layer/h);
y=zeros(predictLen,1);
for i=1:predictLen
    Temp=[];
    for n=1:n_Layer
        if n==1
            noise=Noise(1:resSize(n),i+trainLen);
        else
            noise=Noise(sum(resSize(1:n-1))+1:sum(resSize(1:n)),i+trainLen);
        end
        P1_Input_pattern=Input_streaming(trainLen+i);
        Input_pattern1=times(Input_Mask{n},P1_Input_pattern);
        [State1_n] = layer(Delay_Layer(n),D_state{n}(:,1:2),...
                Input_pattern1,noise,...
                gamma,alpha,h,p);
        Input_pattern_Generator( n,State1_n);
        Temp=[Temp;State1_n];
    end
    y(i)=W_out'*[Temp;0];
end
    
end

