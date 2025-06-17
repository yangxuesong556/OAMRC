%Generate_data_quadratic_memory_task(5000)
clear;
tic
popmin=[0.05,0.1,0.1,1,0,1];
popmax=[3,3,3,10,0,1];  %初始范围    
maxgen=200; %迭代次数
sizepop = 10;  %种群规模100
Vmax = [0.075,0.15,0.15,1.5,1.5,1.5];       %粒子移动最大速度5
Vmin = [-0.075,-0.15,-0.15,-1.5,-1.5,-1.5];      %粒子移动最小速度-5
dim=length(popmax) ;    %测试维度

Max_iter=maxgen;

lb=popmin;%下界
ub=popmax;%上界
%Convergence_curve = zeros(Max_iter,1);%记录最优适应度值
N0=sizepop;%种群数量
pop=initialization(N0,dim,ub,lb);
SalpPositions=pop;
FoodPosition=zeros(1,dim);%食物的位置
FoodFitness=inf;%实物位置

SalpPositions=quzheng(SalpPositions);
SalpFitness=TDR_main(SalpPositions);
%% 


[sorted_salps_fitness,sorted_indexes]=sort(SalpFitness);%排序

for newindex=1:N0
    Sorted_salps(newindex,:)=SalpPositions(sorted_indexes(newindex),:);%重新从小到大排列
end

FoodPosition=Sorted_salps(1,:);%领导者
FoodFitness=sorted_salps_fitness(1);%食物的位置

%% Main loop
%从第二次迭代开始，因为第一次迭代致力于计算 salps 的适应度
l=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness of salps
while l<Max_iter+1
   r(l-1)=0.2*(1-1/(1+exp(-((l-1)-Max_iter/2)*(2/(Max_iter/10)))))+0.4;
    c1=2*(1-1/(1+exp(8-14*(l-1)/Max_iter)));% 2*exp(-(4*l/Max_iter)^2);
   %c1 =2*(1-1/(1+exp(-(l-1-Max_iter/2)*(2/(Max_iter/10)))));%2*exp(-(4*l/Max_iter)^2);% 2*exp(-(4*l/Max_iter)^2);  %Eq. (3.2) in the paper%收敛因子
    
  
   for i=1:size(SalpPositions,1)
        
        SalpPositions= SalpPositions';

        if i<=(N0-2)/2%%领导者  (N0-6)/2round(r(l-1)*(N0-6))
            for j=1:1:dim
                c2=rand();
                c3=rand();
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if c3<0.5 
                    SalpPositions(j,i)=FoodPosition(j)+c1*((ub(j)-lb(j))*c2+lb(j));
                   
                else
                    SalpPositions(j,i)=FoodPosition(j)-c1*((ub(j)-lb(j))*c2+lb(j));
                    
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        elseif i<=N0-2 &&i>(N0-2)/2%round(r(l-1)*(N0-6))
            point1=SalpPositions(:,i-1);
            point2=SalpPositions(:,i);
            SalpPositions(:,i)=(point1+point2)/2;
            
            
%             r0=rand();d=c1*10*(2*rand()-1);
%             alpha=SalpPositions0(:,i-1)-SalpPositions0(:,i);k0=alpha(2)/alpha(1);
%             x=point2(1)+r0*alpha(1);y=point2(2)+r0*alpha(2);
%             SalpPositions(1,i)=x+d;SalpPositions(2,i)=y-d*(-1/k0);
        elseif i<N0+1 &&i>N0-2
            SalpPositions(:,i)=FoodPosition'+c1*cos(rand(dim,1)*pi).*[0.08 0.1 0.1 1 1 1]';%SalpPositions(j,i)=FoodPosition(j)-c1*((ub(j)-lb(j))*c2+lb(j));
            
        end
        SalpPositions= SalpPositions';
        
    end
    
    %%
    for i=1:size(SalpPositions,1)%种群数量  
        
        %调整跑出搜索范围的个体
        Tp=SalpPositions(i,:)>ub;
        Tm=SalpPositions(i,:)<lb;
        SalpPositions(i,:)=(SalpPositions(i,:).*(~(Tp+Tm)))+ub.*Tp+lb.*Tm;
        
        if i<N0+1 &&i>N0-2
            temp=quzheng(SalpPositions(i,:));
            SalpPositions(i,4:6)=temp(4:6);
            SalpFitness(1,i)=TDR_main(SalpPositions(i,:));
        else
            SalpPositions(i,:)=quzheng(SalpPositions(i,:));
            SalpFitness(1,i)=TDR_main(SalpPositions(i,:));
        end
        
        
        
        if SalpFitness(1,i)<FoodFitness
            FoodPosition=SalpPositions(i,:);%更新食物位置
            FoodFitness=SalpFitness(1,i);%更新最优适应值
            
        end

    end

    Convergence_curve(l)=FoodFitness;
    l = l + 1;

%% 可视化+[1,0]
     
    pop=SalpPositions;
    gbest=FoodPosition;
    fprintf('第%d迭代:  \n         最优解:h = %f ,g = %f, a = %f, p=%d, De= %f, La = %f   \n         全局最佳适应度值:%f\n'...
    ,l,gbest(1),gbest(2),gbest(3),gbest(4),gbest(5),gbest(6),FoodFitness);  %输出结果，采取1/fitnessgbest是因为我们设定了值越小，适应度越大。
%
%    figure(1)
%     plot(pop(:,1),pop(:,2),'.k',gbest(1),gbest(2),'rp','MarkerSize',10)
%     hold on
%     plot(SalpPositions(:,1),SalpPositions(:,2),'go')
%     hold off
%     grid on
%     axis([0 popmax(1) 0 popmax(2)])
%     x1=xlabel('x1');
%     x2=ylabel('x2');
%     title(['原始算法：(宏观)迭代次数=' num2str(l-2)]);
%     
% figure(2)
%     plot(pop(:,1),pop(:,2),'.b',gbest(1),gbest(2),'rp','MarkerSize',10)
%     pause(0.2)%暂停程序运行0.1s
%     x1=xlabel('x1');
%     x2=ylabel('x2');
%     title(['SSA_leader算法：(微观)迭代次数=' num2str(l-2)]);
%     drawnow;%刷新屏幕
%     frame = getframe();%捕获影片帧
%     im = frame2im(frame);
%     [A,map] = rgb2ind(im,256);%将真彩图像转换为索引图像
%     if l == 1-2
%         imwrite(A,map,'F:\0.1\基于樽海鞘算法的图像匹配\粒子群动态图像\标准PSO.gif','gif','LoopCount',Inf,'DelayTime',0.1);
%     else
%         imwrite(A,map,'F:\0.1\基于樽海鞘算法的图像匹配\粒子群动态图像\标准PSO.gif','gif','WriteMode','append','DelayTime',0.1);
%     end
end
T=l-1;
 figure(2)
 plot(Convergence_curve(1:end))
toc