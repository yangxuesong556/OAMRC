function [State]=train_TDR(TPD,Input_streaming,Noise,Input_Mask,...
    gamma,alpha,h,p,Delay_Layer,Delay_time)
resSize=round(Delay_Layer/h);
trainLen=TPD(1);
n_Layer=length(Delay_Layer); %储备池层数
global D_state;
D_state=cell(n_Layer,1);
for i=1:n_Layer
        D_state{i}=zeros(resSize(i),2);
end
State=zeros(sum(resSize)+1,trainLen);

for i=1:trainLen%9
    
    Temp=[];
    for n=1:n_Layer
        if n==1
            noise=Noise(1:resSize(n),i);
        else
            noise=Noise(sum(resSize(1:n-1))+1:sum(resSize(1:n)),i);
        end

        Input_pattern1=times(Input_Mask{n},Input_streaming(i));
        [State1_n] = layer(Delay_Layer(n),D_state{n}(:,1:2),...
                Input_pattern1,noise,...
                gamma,alpha,h,p);
        Input_pattern_Generator( n,State1_n);
        Temp=[Temp;State1_n];
    end
    State(:,i)=[Temp;0];
end
end
