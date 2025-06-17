function [State]=train_TDR(TPD,Input_streaming,Noise,Input_Mask,...
    gamma,alpha,h,p,...
    Delay_Layer,Delay_time)

global D_state;
global L_state;
resSize=round(Delay_Layer/h);
trainLen=TPD(1);
n_Layer=length(Delay_Layer); %储备池层数
D_state=cell(n_Layer-1,1);
L_state=cell(n_Layer-1,1);
for i=1:n_Layer
    if i==1
        State1_n_1=zeros(resSize(i),1);
    else
        D_state{i-1}=zeros(resSize(i),sum(Delay_time(1:i-1))+1);
    end
        L_state{i}=zeros(resSize(i),1);
end

% Input=Input_streaming(1);
State=zeros(sum(resSize)+1,trainLen);
Temp=zeros(sum(resSize),1);

for i=1:trainLen%9
    
    Temp=0*Temp;
    for n=1:n_Layer
        if n==1
            Input_pattern1=times(Input_Mask{n},Input_streaming(i));
            [State1_n] = layer(Delay_Layer(n),State1_n_1,L_state{n}(:,1),...
                Input_pattern1,Noise(1:resSize(n),i),...
                gamma,alpha,h,p);
            Temp(1:resSize(n),1)=State1_n;
            State1_n_1=State1_n;
            L_state{n}(:,1)=State1_n;
        else
            Input_pattern_Generator( n,State1_n);
            Input_pattern2=times(Input_Mask{n},Input_streaming(i));
            [State1_n] = layer(Delay_Layer(n),D_state{n-1}(:,1),L_state{n}(:,1),... %+1D_state{n-1}(:,sum(Delay_time(1:n-1))+1:end)
                Input_pattern2,Noise(sum(resSize(1:n-1))+1:sum(resSize(1:n)),i),...
                gamma,alpha,h,p);
            % 第二层开始 D_State 储存的是延迟的状态，L_State储存的是上一个时刻各层的状态，
            % 这里传的是最后延时的状态，L_State传的是 该层上个时刻的状态
            Temp(sum(resSize(1:n-1))+1:sum(resSize(1:n)),1)=State1_n;
            L_state{n}(:,1)=State1_n;
            %State1_n_1{n}=Input_pattern_Generator(n,State1_n_1{n},State2_n);
        end
    end
    %State(:,i)=[Temp;Input_streaming(i)];
    State(:,i)=[Temp;0];
end
end