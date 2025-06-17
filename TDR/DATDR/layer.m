function X_State = layer(delay_time,State,LState,Input_Masked,noise,gamma,alpha,h,p)
N=size(State,1);
% �������ľ�����ʱ���״̬
x=[State,zeros(N,1)]; %����һ�� ��202�� 200*202
t=size(x,2); % 202
fx=@(xh,J,z,a,g,p0)(a*(xh+g*J+z)/(1+(xh+g*J+z)^p0));
% ������������
for j=1:N 
    if j==1
        xh=x(j,t-1);
        %xh ���� ��������״̬ ����ʱ���״̬
        x(j,t)=1/(1+h)*LState(end,1)+h/(1+h) *fx(xh,Input_Masked(j),noise(j),alpha,gamma,p);%x(N,t-1)
     % L_State �����ϸ�ʱ�̵�״̬��LState(end,1)��ʾ����һ��ʱ�� �˲� ���һ������ڵ��ֵ
    elseif j>1&&j<=N
        xh=x(j,t-1);
        x(j,t)=1/(1+h)*x(j-1,t)+h/(1+h) *fx(xh,Input_Masked(j),noise(j),alpha,gamma,p);
   % ����ʽ�� ��ʾ��һ������ڵ� ������һ��
   % ֻ�ܴ� x(j,t)=1/(1+h)*x(j-1,t) ��������     
     end
end
X_State =x(:,end);
end
%% 