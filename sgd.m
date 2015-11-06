function sgd(stepsize, points)
    close all ; 
    load('data1.mat');
    tol = exp(-3);
    lambda= exp(-2);
    kernel = rbf_kernel(TrainingX);
    prevtheta = ones(size(TrainingX,1),1);
    theta = ones(size(TrainingX,1),1);
    curiter = 0;
    shuffle=1:1:size(kernel,1);
    shuffle=(randperm(size(kernel,1)));
    points_for_gd = shuffle(1:points);
    time_start = tic;
    all_time(1) = time_start;
    cost(1) = get_cost(TrainingY,prevtheta,kernel,lambda);
    %S_G = ggradient(TrainingY(points_for_gd(1:points)), prevtheta, lambda, kernel(points_for_gd(1:points)));
    %theta = prevtheta - stepsize.*S_G;
    %diff = norm(theta - prevtheta(points_for_gd(1:points)))+tol
    diff=tol;
    while diff >= tol 
        prevtheta = theta;
        G = ggradient(TrainingY(points_for_gd(1:points)), prevtheta, lambda, kernel(points_for_gd(1:points),:));
        theta = prevtheta - stepsize.*G;
        time_now = toc(time_start);
        diff = norm(theta -prevtheta);
        curiter = curiter + 1;
        all_time(curiter) = time_now;
        cost(curiter) = get_cost(TrainingY,prevtheta(points_for_gd(1:points)),kernel,lambda);
        accuracy(curiter) = error(prevtheta,kernel(points_for_gd(1:points)),TestY(points_for_gd(1:points)));
        shuffle=1:1:size(kernel,1);
        shuffle=(randperm(size(kernel,1)));
        points_for_gd = shuffle(1:points);
    end
    fprintf ('Accuracy:%0.4f, step:%0.4f, iter:%d, total time: %d  \n' , accuracy(end), stepsize, curiter, all_time(end) );
end

function cost = get_cost(y,theta,kernel,lambda)

    N=size(y,1);
    total = 0;
    for i=1:N
        u = y(i)*theta'*kernel(:,i);
        size(u);
        s =1./(1+exp(-u));
        total = total + log(s);
    end
    
    cost = -total./N + lambda*(theta'*theta);
    
end

function g = ggradient(y, theta, lambda, kernel)
    
    N = size(y,1);
    total=zeros(1,N);
    last = 2*lambda*theta';
    size(kernel(:,1))
    size(theta)
    for i=1:N
        u = y(i)*theta'*kernel(:,i);
        s = 1./(1+exp(-u));
        ss = ((1-s)*y(i)*kernel(:,i)');
        size(total)
        size(ss)
        total = total + ss;
    end
    g = -total./N + last;
    g=g';
    
end

function t_error = error(theta,kernel,Y)

    N = size(kernel,1);
    err = zeros(N,1);
    for i=1:N
        prob = 1./(1+ exp(-theta'*kernel(:,i)));
        if (prob >=0.5)
            err(i)=1;
        else
            err(i)=-1;
        end
    end
    sum(err~=Y);
    t_error = sum(err~=Y)/length(Y);
end

function K = rbf_kernel(X)
% Inputs:
%       X:      data matrix with training samples in rows and features in columns
% Output:
%       K: kernel matrix

    N = size(X,1);
    D=squareform(pdist(X,'euclidean')).^2;
    K = exp(-D./(1/(N^2))*sum(D(:)));
end