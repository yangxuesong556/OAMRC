function [ ] = Generate_data_NARMA20_task(T)
u=unifrnd(0,0.5,[1,T+30]);
y=[unifrnd(0,0.5,[1,30]),zeros(1,T)];
for i=30:T+29
    
    y(i+1)=tanh(0.3*y(i)+0.05*y(i)*sum(y(:,i-19:i))+1.5*u(i)*u(i-19)+0.01);

end
u=u(31:end);
y=y(31:end);
save('C:\Users\17519\Desktop\paper文件\第三篇\GitHub\第三篇代码 - 第二版\narma10-30修\TDR\DATDR\数据集\NARMA20data.mat','u','y')
end

