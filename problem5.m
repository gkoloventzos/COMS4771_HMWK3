function problem5()
    close all ; 
    load('data1.mat') ;
    stepsize = 2;
    tol = 1e-2;
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
        r = risk(TestX, TestY, theta );
        f = 1./(1+exp(-TestX*theta));
        f(f >= 0.5) = 1;
        f(f < 0.5) = -1;
        err = sum(f~=Y) / length (Y) ;
        %fprintf ( 'Iter:%d , error:%0.4f , risk:%0.4f \n' , curiter , err , r );
        risks = cat(1, risks , r );
        errs = cat(1, errs , err );
        % Update theta
        prevtheta = theta ;
        G = gradient (TestX, TestY, theta );
        theta = theta - stepsize*G;
        curiter = curiter + 1;
    end
    figure , plot (1: curiter , errs , 'r' , 1:curiter , risks , 'g' );
    title('Error ( red ) and risk ( green ) vs . iterations ' );
    disp('theta');
    disp(theta)
    x=0:0.01:1;
    y=(-theta(3) - theta (1).* x)/ theta (2);
    figure , plot (x, y, 'r' ); hold on ;
    plot (TestX(: , 1 ) , TestX(: , 2 ) , '.' ) ;
    title ( 'Linear decision boundary ' );
end

function R = risk (x, y, theta )
    f = 1./(1+exp(-x* theta ));
    r = (y-1).*log(1-f)-y.*log(f); r(isnan(r)) = 0;
    R = mean(r);
end
    
function g = gradient (x, y, theta )
    yy = repmat (y , 1 , size (x,2));
    f = 1./(1+exp(-x*theta ));
    ff = repmat ( f , 1 , size (x,2));
    d = x.*repmat (exp(-x*theta ) , 1, size (x,2));
    g = (1-yy ).*( x-d.* ff) - yy.* d.* ff ;
    g = sum(g);
    g = g/length (y);
    g = g';
end

function e=myexp(x,theta)
    k_2= 
end