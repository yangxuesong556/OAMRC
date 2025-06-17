function [ State ] = Train_UpStates( Input_streaming,Training_data,TPW,W_in,W,W_Feedb,resSize )
%UNTITLED8 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%% this is the train stage function. 
global D_state;%D��״̬
global Delay;
global Leaky_rate;%й¶ϵ��
trainLen=TPW(1);
n_layers=length(resSize);
State1_n_1=cell(n_layers,1);%��������ǰһ��״̬��ֵ������Ԫ������n*1,���ԃ���ÿ��
State1=cell(n_layers,1);%��������Ŀǰ״̬��ֵ������Ԫ������n*1
for i=1:n_layers % ��������ѵ��ģ�飬��˶�ÿһ��Ĵ������ڲ�״̬����һ��reSize*trainLen�ľ���������100*3200�ľ���
    State1 {i}=zeros(resSize(i),trainLen);
    State1_n_1{i}=State1{i}(:,1);%����һ��״̬�����ݵ�ֵ������һ��״̬
    D_state{i}=zeros(resSize(i)+1,sum(Delay(1:i)));  %�������������Ӵ���֮������ݣ����кܶ������ݽ��� D_state is used to temporarily store the state of the reservoir between adjacent layers. delay�൱��flag��¼�ڼ������ʱģ��
end%������Ҫ�Ƕ�ѵ��ģ������ݽ������ݴ洢
Output_pattern=0;%���������Ϊ0
State=zeros(sum(resSize)+1,trainLen);
Temp=zeros(sum(resSize),1);%���󷵻�
for i=1:trainLen %�Դ����ص�״̬���и���
    Temp=0*Temp; %temp����
    for n=1:n_layers %��2��ʱ�Ͳ�һ����
        if n==1  %����һ��ʱ�����ʱ���ڲ�ѵ�����ݵ�ģ�͸���ͳ��RC��ͬ
            Input_pattern1=Input_streaming(i);
            [State1_n] = Update(State1_n_1{n},Input_pattern1,Output_pattern,W_in{n},W{n},W_Feedb{n});
             %tan��Input_pattern.* W_in+W*States_n_1+Output_pattern*W_Feedb��
            State1_n_1{n}=(1-Leaky_rate).*State1_n_1{n}+Leaky_rate.*State1_n;%�õ��ڲ�״̬���µ�ֵ
            Temp(1:resSize(n),1)=State1_n_1{n};%���ڲ�״̬���浽Temp��
            State2_next=Input_pattern_Generator(n,State1_n_1{n},Input_pattern1);%state2_n��������� ���º���ڲ�״̬�Լ���������ݣ�Ϊһ��1001*1�ľ���
        else
            Input_pattern2=State2_next; %��һ�������Ϊ��һ��״̬�������Լ���һ�������ݣ� ����Ϊ���ε����룬���ڶ��㿪ʼ   
            [State2_n]= Deep_Update(State1_n_1{n},Input_pattern2,Output_pattern,W_in{n},W{n},W_Feedb{n}); %ʵ������һ���ľ����������ݱ�Ϊ��һ�����ݵ����
            State1_n_1{n}=(1-Leaky_rate).*State1_n_1{n}+Leaky_rate.*State2_n;
            State2_next=Input_pattern_Generator(n,State1_n_1{n},Input_pattern1); %���ڲ�״̬���ݴ洢����һ��״̬�洢���У�Ȼ�����
            Temp(sum(resSize(1:n-1))+1:sum(resSize(1:n)),1)=State1_n_1{n};%���ݴ���,����
        end
    end
    Output_pattern=Training_data(i);
     State(:,i)=[Temp;0];
end
end
    

