function [ State2_n ] = Input_pattern_Generator(n,State1_n,Input_pattern1)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明，由于不是简单的数据输入，而是分为很多个子层，第一层的输入是由数据直接进行输入
%而第二层的输入就不是简单的顺序输入而是要考虑上一层的输出以及延时
%% Control delay
global D_state;%对延时的状态以及延时进行定义
global Delay;
if sum(find(Delay==0)==n)%find线性索引非零元素 当Delay向量中的延时为0时，将上一个状态和输入列入变量中，这里是为了区分 delay的存在与传统的ESN做出对比
    State2_n=[State1_n;Input_pattern1];%就是当n=1时，state2_n里面的数据为 更新后的状态以及输入的数据
else
    State2_n=D_state{n}(:,end);%表示取最后一列的数据赋值给State_n，由于可能有多个数据进行存储，因此取外层的即从第一层传来的数据
    D_state{n}(:,2:end)=D_state{n}(:,1:end-1);%D_state进行列数据前移
    D_state{n}(:,1)=[State1_n;Input_pattern1];%将更新后的状态以及输入的数据置于延时模块的第一列
end%这里总的来说就是对于Dealy模块的控制，取前一个的输入作为此时的输出以及将上一个的输出进行存储
end
%思考：delay表示几个时刻的延时，如何将延时进行下去。
