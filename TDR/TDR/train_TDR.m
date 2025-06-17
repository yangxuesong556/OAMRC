function [State]=train_TDR(TPD,Input_streaming,Noise,Input_Mask,...
    gamma,alpha,h,p,Delay_Layer,Delay_time)
resSize=round(Delay_Layer/h);
trainLen=TPD(1);
State1_n_1=zeros(resSize,2);
State=zeros(sum(resSize)+1,trainLen);
Temp=zeros(sum(resSize),1);

for i=1:trainLen%9
    Temp=0*Temp;
    n=1;
    Input_pattern1=times(Input_Mask{n},Input_streaming(i));
    [State1_n] = layer(Delay_Layer(n),State1_n_1,...
        Input_pattern1,Noise(1:resSize,i),...
        gamma,alpha,h,p);
    Temp(1:resSize,1)=State1_n;
    State1_n_1=[State1_n_1(:,2),State1_n];
    State(:,i)=[Temp;0];
end
%State(:,i)=[Temp;Input_streaming(i)];

end
