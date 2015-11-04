function problem5()
    close all ; 
    load('data1.mat');
    stepsize = 2;
    tol = 1e-2;
    lambda= 1e-2;
    theta = rand(size(TestX, 2), 1 );
    maxiter = 200000;
    curiter = 0;
    % For plotting stats
    risks = [];
    errs = [];
    prevtheta = theta+2*tol;
    while norm(theta - prevtheta) >= tol
        if curiter > maxiter
            break ;
        end
        % Current stats
        kernel = rbf_kernel(TrainingX,TestX,size(TestX,1))
        r = risk(TestX, TestY, theta );
        f = 1./(1+exp(-TestX*theta));
        f(f >= 0.5) = 1;
        f(f < 0.5) = -1;
        err = sum(f~=TestY)/length(TestY) ;
        %fprintf ( 'Iter:%d , error:%0.4f , risk:%0.4f \n' , curiter , err , r );
        %risks = cat(1, risks , r );
        errs = cat(1, errs , err );
        % Update theta
        prevtheta = theta ;
        G = gradient (TestX, TestY, theta, kernel);
        theta = theta - stepsize*G;
        curiter = curiter + 1;
    end
    %figure , plot (1: curiter , errs , 'r' , 1:curiter , risks , 'g' );
    %title('Error ( red ) and risk ( green ) vs . iterations ' );
    %disp('theta');
    %disp(theta)
    x=0:0.01:1;
    y=(-theta(3) - theta (1).* x)/ theta (2);
    %figure , plot (x, y, 'r' ); hold on ;
    %plot (TestX(: , 1 ) , TestX(: , 2 ) , '.' ) ;
    %title ( 'Linear decision boundary ' );
end

function R = risk (x, y, theta )
    f = 1./(1+exp(-x* theta ));
    r = (y-1).*log(1-f)-y.*log(f); r(isnan(r)) = 0;
    R = mean(r);
end
    
function g = gradient (x, y, theta, lambda)

    g = -(1/size(x,1))*1-(1/1+exp(y*theta'*kernel)*y*kernel') + 2*lambda*theta';

end

%function th=myexp(x,theta)
%    th=0;
%end

function K = rbf_kernel(X,X2,sigma)
% Inputs:
%       X:      data matrix with training samples in rows and features in columns
%       X2:     data matrix with test samples in rows and features in columns
%       sigma: width of the RBF kernel
% Output:
%       K: kernel matrix

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end;
        K = exp(-D/(2*sigma^2));
end