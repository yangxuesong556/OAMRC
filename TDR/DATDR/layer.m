function X_State = layer(delay_time,State,LState,Input_Masked,noise,gamma,alpha,h,p)
N=size(State,1);
% 传进来的就是延时后的状态
x=[State,zeros(N,1)]; %加了一列 共202列 200*202
t=size(x,2); % 202
fx=@(xh,J,z,a,g,p0)(a*(xh+g*J+z)/(1+(xh+g*J+z)^p0));
% 定义匿名函数
for j=1:N 
    if j==1
        xh=x(j,t-1);
        %xh 储存 传进来的状态 即延时后的状态
        x(j,t)=1/(1+h)*LState(end,1)+h/(1+h) *fx(xh,Input_Masked(j),noise(j),alpha,gamma,p);%x(N,t-1)
     % L_State 代表上个时刻的状态，LState(end,1)表示接上一个时刻 此层 最后一个虚拟节点的值
    elseif j>1&&j<=N
        xh=x(j,t-1);
        x(j,t)=1/(1+h)*x(j-1,t)+h/(1+h) *fx(xh,Input_Masked(j),noise(j),alpha,gamma,p);
   % 按公式来 表示上一个虚拟节点 传到下一个
   % 只能从 x(j,t)=1/(1+h)*x(j-1,t) 来做文章     
     end
end
X_State =x(:,end);
end
%% 