function [ State ] = Train_UpStates( Input_streaming,Training_data,TPW,W_in,W,W_Feedb,resSize )
%UNTITLED8 此处显示有关此函数的摘要
%   此处显示详细说明
%% this is the train stage function. 
global D_state;%D的状态
global Delay;
global Leaky_rate;%泄露系数
trainLen=TPW(1);
n_layers=length(resSize);
State1_n_1=cell(n_layers,1);%储备池中前一个状态的值，生成元胞数组n*1,可以儲存每個
State1=cell(n_layers,1);%储备池中目前状态的值，生成元胞数组n*1
for i=1:n_layers % 由于这是训练模块，因此对每一层的储备池内部状态赋予一个reSize*trainLen的矩阵这里是100*3200的矩阵
    State1 {i}=zeros(resSize(i),trainLen);
    State1_n_1{i}=State1{i}(:,1);%将上一个状态的数据的值赋予下一个状态
    D_state{i}=zeros(resSize(i)+1,sum(Delay(1:i)));  %用来储存两个子储层之间的数据，会有很多层的数据进行 D_state is used to temporarily store the state of the reservoir between adjacent layers. delay相当于flag记录第几层的延时模块
end%这里主要是对训练模块的数据进行数据存储
Output_pattern=0;%输出数据置为0
State=zeros(sum(resSize)+1,trainLen);
Temp=zeros(sum(resSize),1);%矩阵返回
for i=1:trainLen %对储备池的状态进行更新
    Temp=0*Temp; %temp更新
    for n=1:n_layers %有2层时就不一样了
        if n==1  %仅有一层时，则此时的内部训练数据的模型跟传统的RC相同
            Input_pattern1=Input_streaming(i);
            [State1_n] = Update(State1_n_1{n},Input_pattern1,Output_pattern,W_in{n},W{n},W_Feedb{n});
             %tan（Input_pattern.* W_in+W*States_n_1+Output_pattern*W_Feedb）
            State1_n_1{n}=(1-Leaky_rate).*State1_n_1{n}+Leaky_rate.*State1_n;%得到内部状态更新的值
            Temp(1:resSize(n),1)=State1_n_1{n};%将内部状态储存到Temp中
            State2_next=Input_pattern_Generator(n,State1_n_1{n},Input_pattern1);%state2_n里面的数据 更新后的内部状态以及输入的数据，为一个1001*1的矩阵
        else
            Input_pattern2=State2_next; %下一层的输入为上一个状态的输入以及上一个的数据， 这里为深层次的输入，即第二层开始   
            [State2_n]= Deep_Update(State1_n_1{n},Input_pattern2,Output_pattern,W_in{n},W{n},W_Feedb{n}); %实际上是一样的就是输入数据变为上一个数据的输出
            State1_n_1{n}=(1-Leaky_rate).*State1_n_1{n}+Leaky_rate.*State2_n;
            State2_next=Input_pattern_Generator(n,State1_n_1{n},Input_pattern1); %将内部状态数据存储到下一个状态存储当中，然后对于
            Temp(sum(resSize(1:n-1))+1:sum(resSize(1:n)),1)=State1_n_1{n};%数据存入,对于
        end
    end
    Output_pattern=Training_data(i);
     State(:,i)=[Temp;0];
end
end
    

