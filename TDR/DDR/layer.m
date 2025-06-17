function X_State = layer(delay_time,State,Input_Masked,noise,gamma,alpha,h,p)
N=size(State,1);
x=[State,zeros(N,1)]; %加了一列 共202列 200*202
t=size(x,2); % 202
fx=@(xh,J,z,a,g,p0)(a*(xh+g*J+z)/(1+(xh+g*J+z)^p0));
for j=1:N 
    if j==1
        xh=x(N,t-2);
        x(j,t)=1/(1+h)*x(N,t-1)+h/(1+h) *fx(xh,Input_Masked(j),noise(j),alpha,gamma,p);

    elseif j>1&&j<=N
        xh=x(j-1,t-1);
        x(j,t)=1/(1+h)*x(j-1,t)+h/(1+h) *fx(xh,Input_Masked(j),noise(j),alpha,gamma,p);

     end
end
X_State =x(:,end);
end
