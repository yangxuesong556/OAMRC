function [ ] = Generate_data_NARMA10_task(T)
u=unifrnd(0,0.5,[1,T+30]);
y=[unifrnd(0,0.5,[1,30]),zeros(1,T)];
for i=30:T+29
    
    y(i+1)=0.3*y(i)+0.05*y(i)*sum(y(:,i-9:i))+1.5*u(i)*u(i-9)+0.1;

end
u=u(31:end);
y=y(31:end);
save('C:\Users\17519\Desktop\第三篇代码\narma10-30修\TDR\DATDR\数据集\NARMA10data.mat','u','y')
end

