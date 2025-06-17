clear;
tic
popmin=[0.05,0.9,0.9,0,1];
popmax=[1,1,1,0,1];  %初始范围    
maxgen=50; %迭代次数
sizepop = 20;  %种群规模100
Vmax = [0.075,0.15,0.15,1.5,1.5];       %粒子移动最大速度5
Vmin = [-0.075,-0.15,-0.15,-1.5,-1.5];      %粒子移动最小速度-5
Dim=length(popmax) ;    %测试维度

wmax=0.9;wmin=0.4;
W=wmax*ones(1,sizepop);
c1 =1.5;       %认知权重因子1.5
c2 =1.5;       %社会权重因子1.5
c3=0;

re=sizepop*0.6;%剩余种群规模
record=zeros(maxgen,Dim);

%立方混沌映射
IN=2*rand(1,Dim)-ones(1,Dim);
inpop=[IN;zeros(sizepop-1,Dim)];
for i=1:sizepop-1
    inpop(i+1,:)=4*inpop(i,:).^3-3*inpop(i,:);
end

%映射到可行域坐标
pop=zeros(sizepop,Dim);
for i=1:sizepop
   pop(i,:)=popmin+(ones(1,Dim)+inpop(i,:)).*(popmax-popmin)/2;
    
end
%pop(:,end)=round(pop(:,end))
pop=quzheng(pop);

for i = 1:sizepop
 V(i,:) = (Vmax-Vmin).*rand(1,Dim)+Vmin;       %初始化速度
    % 计算适应度
    %[fitness(i),RMSE(i,:),NRMSE(i,:)] = RC_main(pop(i,:),20);   %计算适应度(这里设定的是结果的值越小=适应度越低)
end
fitness=TDR_main(pop);% 计算适应度

toc
pop
fitness0=fitness;%保留上一次适应度值的数据

[bestfitness bestindex] = min(fitness); %bestindex:全局最优粒子索引
gbest = pop(bestindex,:);   %全局最佳位置
pbest = pop;    %个体最佳
fitnesspbest = fitness;   %个体最佳适应度值
fitnessgbest = bestfitness;   %全局最佳适应度值
record(1,:)=gbest;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% 迭代
a00=zeros(1,maxgen);
for i = 1:maxgen       %代数更迭
    for j = 1:sizepop  %遍历个体
        % 速度更新W(j)
        V(j,:) = 0.5*V(j,:) + c1*rand(1,Dim).*(pbest(j,:) - pop(j,:)) + c2*rand(1,Dim).*(gbest - pop(j,:));
        %速度边界处理
        V(j,find(V(j,:)>Vmax)) = Vmax(find(V(j,:)>Vmax));
        V(j,find(V(j,:)<Vmin)) = Vmin(find(V(j,:)<Vmin));
        
        % 种群更新
        pop(j,:) = pop(j,:) + V(j,:);
        %位置边界处理
        
        pop(j,find(pop(j,:)>popmax)) = popmax(find(pop(j,:)>popmax));
        pop(j,find(pop(j,:)<popmin)) = popmin(find(pop(j,:)<popmin));
    end
   pop=quzheng(pop);
   fitness=TDR_main(pop);% 计算适应度

%  %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% %加入扰动策略、判断过早收敛
%     if max(sqrt((fitness-mean(fitness)*ones(1,sizepop)).^2))>1
%         f=max(sqrt((fitness-mean(fitness)*ones(1,sizepop)).^2));
%     else
%         f=1;
%     end
%     sigma2=0;
%     for i1=1:sizepop
%         sigma2=sigma2+((fitness(i1)-mean(fitness))/f)^2;
%     end
%         %sigma2=sum(((fitness-mean(fitness))./f).^2)
%     sigma2;
% %立方混沌映射%跳出局部最优点
%     if sigma2<1
%         a00(i)=i;
%         REIN=2*rand(1,Dim)-ones(1,Dim);
%         reinpop=[REIN;zeros(re-1,Dim)];
%         for i1=1:re-1
%             reinpop(i1+1,:)=4*reinpop(i1,:).^3-3*reinpop(i1,:);
%         end
% 
%         %映射到可行域坐标
%         repop=popmin*ones(re,Dim)+(ones(re,Dim)+reinpop).*(popmax-popmin)/2;
%         %计算适应度值
%         for i1 = 1:re
%             refitness(i1) = fun(repop(i1,:),index);   %计算适应度(这里设定的是结果的值越小=适应度越低)
%         end
% %替换
%         [veal,num]=sort(fitness);
%         for i1=sizepop-re+1:sizepop
%                 pop(num(i1),:)=repop(i1-(sizepop-re),:);
%         end   
%     end
%    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%更新    
        for j = 1:sizepop
        % 个体最优更新
            if fitness(j) < fitnesspbest(j)
                pbest(j,:) = pop(j,:);
                fitnesspbest(j) = fitness(j);
            end
            % 群体最优更新
            if fitness(j) < fitnessgbest
               gbest = pop(j,:);
            fitnessgbest = fitness(j);
            end
        end


%惯性权重的自适应变化
%     K=(fitness-fitness0)./fitness0;%粒子相对变化率
%     W=(ones(1,sizepop)+exp(-1.*K)).^(-1);%惯性权重调整公式
 
    fitness0=fitness;
    
    fprintf('第%d迭代:  \n         最优解:h = %f ,g = %f, a = %f, De= %f, La = %f \n         全局最佳适应度值:%f\n'...
     ,i,gbest(1),gbest(2),gbest(3),gbest(4),gbest(5),fitnessgbest);  %输出结果，采取1/fitnessgbest是因为我们设定了值越小，适应度越大。
 F(i)=fitnessgbest;
 %pop
  record(i+1,:)=gbest;
  toc
end
 F1=F;
figure(2)
plot(1:length(F1),F1)
title('原始算法：适应度变化曲线');
fitnessgbest
find(a00~=0)
