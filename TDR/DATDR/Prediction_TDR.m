function [ y] = Prediction_TDR(State0,Input_streaming,TPD,Noise,Input_Mask,W_out,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
global D_state;
global L_state;
trainLen=TPD(1);
predictLen=TPD(2);
n_layers=length(Delay_Layer);
resSize=round(Delay_Layer/h);
y=zeros(predictLen,1);

Temp=zeros(sum(resSize),1);
State1_n_1=State0(1:resSize(1),:);
for i=1:predictLen
    
    Temp=0*Temp;
    P1_Input_pattern=Input_streaming(trainLen+i);
    for n=1:n_layers
        if n==1
            
            Input_pattern1=times(Input_Mask{n},P1_Input_pattern);
            [State1_n] = layer(Delay_Layer(n),State1_n_1,L_state{n}(:,1),...
                Input_pattern1,Noise(1:resSize(n),trainLen+i),...
                gamma,alpha,h,p);

            Temp(1:resSize(n),1)=State1_n;
            State1_n_1=State1_n;
            L_state{n}(:,1)=State1_n;
        else
            Input_pattern_Generator(n,State1_n);
            Input_pattern2=times(Input_Mask{n},P1_Input_pattern);
            [State1_n] = layer(Delay_Layer(n),D_state{n-1}(:,1),L_state{n}(:,1),...  %+1D_state{n-1}(:,sum(Delay_time(1:n-1))+1:end)
                Input_pattern2,Noise(sum(resSize(1:n-1))+1:sum(resSize(1:n)),trainLen+i),...
                gamma,alpha,h,p);   
            Temp(sum(resSize(1:n-1))+1:sum(resSize(1:n)),1)=State1_n; 
            L_state{n}(:,1)=State1_n;
        end
        
    end
    y(i)=W_out'*[Temp;0];
end

end