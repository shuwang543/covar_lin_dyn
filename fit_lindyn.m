function [A_fit,ss_fit,Sigma_0_list,mu_0_list,lossval,output2] = fit_lindyn(mvals,Cvals,tvals,error_model,edu_guess_flag,alpha,reps,guess_scale,stream)
    % mvals: cell array of means, Cvals: cell array of covariances, 
    % tvals: array of time values, error_model: model of Measurement Error
    % (e.g. 0 means no error, sigma*eye(n) gives diagonal matrix of error)
    % edu_guess_flag: 0/1 to indicate whether to use the theory-guided initialization
    % alpha: weight of the mean terms in the loss-function relative to the
    % covariance terms
    % alpha: how much to weight the means-loss relative to the covariance-los
    % reps: how many initializations to try for optimization
    % guess_scale: factor to multiply all entries of random initializations
    % (at 1, random initializations are chosen with entries between -0.5 to
    % 0.5)
    % stream: variable for setting random seeds for high performance
    % computing
    n_var = size(Cvals{1},1);
    
    %% example of loss function where everything that can be fixed is fixed, and for a single time-series only
    warning('off','MATLAB:logm:nonPosRealEig')
    
    Sigma_fix = Cvals{1};
    mu_fix = mvals{1};
    
    %parms will be a list that is (n^2 + n) long, for A and x_guess
    idx_A = 1:n_var^2;
    idx_ss = (n_var^2+1):(n_var^2 + n_var);

    options = optimoptions('fminunc','Display','off','MaxFunctionEvaluations',10^6,...
    'OptimalityTolerance',1e-7,'MaxIterations',10^6,...
    'FiniteDifferenceStepSize',1e-7,'StepTolerance',1e-10,'FiniteDifferenceType','central');%,...
    %'PlotFcn','optimplotfval');
    
    flag_test=0;
    lossfn = @(parms) lossfn_ser_covar(parms(idx_A),Sigma_fix,tvals,Cvals,error_model) + alpha*lossfn_ser_mu(parms(idx_A),mu_fix,parms(idx_ss),tvals,mvals);
    if edu_guess_flag==0
        while(flag_test==0)
            A_guess = guess_scale*(rand(stream,n_var)-0.5);
            ss_guess = guess_scale*(rand(stream,n_var,1)-0.5);
            try
                init_loss = feval(lossfn,[A_guess(:);ss_guess(:)]);
                if(imag(init_loss)==0)
                    flag_test=1;
                end
            catch ME
                disp(ME.message)
                continue
            end
        end
    elseif edu_guess_flag==1
        [A_guess,~] = eduguess_A(Cvals,mvals,mean(diff(tvals)));
        ss_guess = guess_scale*(rand(stream,n_var,1)-0.5);
    else
        [A_guess,ss_guess] = eduguess_A(Cvals,mvals,mean(diff(tvals)));
    end 
    
    if rcond(A_guess) < 1e-10
        return
    end
   

    
    parms_fit = cell(reps,1);
    lossvals = zeros(reps,1);
    outputs = cell(reps,1);
    for i = 1:reps
        [parms_fit{i},lossvals(i),~,outputs{i}] = fminunc(lossfn,[A_guess(:);ss_guess],options);
    end
    [~,idx_fit] = min(lossvals);
    parm_fit = parms_fit{idx_fit};
    lossval = lossvals(idx_fit);
    output2 = outputs{idx_fit};

    A_fit = reshape(parm_fit(idx_A),[n_var,n_var]);
    ss_fit = parm_fit(idx_ss);
    Sigma_0_list = Cvals{1};
    mu_0_list = mvals{1};

    t_temp = min(diff(tvals));
    V_fit = expm(t_temp*A_fit);
    A_fit = logm(V_fit)/t_temp;

end

%% base function for the mean & covariance losses
    
function L_ser_cov = lossfn_ser_covar(A,Sigma,tvals,Cvals,noise_model)
    %A is the Jacobian of the dynamical system
    %sigmavec is the initial distribution's covariance matrix's upper
    %triangular entries
    %tvals is the time-values of each sampled distribution
    %Qvals are the covariance matrices at each time point
    %noise is a covariance matrix modeling measurement noise
    n_var = size(Cvals{1},1);
    A = reshape(A,[n_var,n_var]);
    Sigma = triu(Sigma) + triu(Sigma,1)'; %force Sigma to be symmetric
    A_tvals = arrayfun(@(t) t*A,tvals,'UniformOutput',false);
    V_tvals = cellfun(@expm, A_tvals,'UniformOutput',false);
    distfun = @(V_t,Q,Sig,Err) sum(log(eig(V_t*Sig*V_t'+Err,Q)).^2);
    tempfun = @(V_t,Q) distfun(V_t,Q,Sigma,noise_model);
    
    L_ser_cov = sum(cell2mat(cellfun(tempfun, V_tvals, Cvals,'UniformOutput',false)));
end

function L_ser_mu = lossfn_ser_mu(A,mu_0,x_ss,tvals,mvals)
    %the mean vectors are formatted as 1 x n vectors
    n_var = numel(mvals{1});
    A = reshape(A,[n_var,n_var]);
    A_tvals = arrayfun(@(t) t*A',tvals,'UniformOutput',false);
    V_tvals = cellfun(@expm, A_tvals,'UniformOutput',false);
    pred_mu = cellfun(@mtimes,repmat({mu_0-x_ss'},1,numel(tvals)),V_tvals','UniformOutput',false);
    L_ser_mu = sum(cellfun(@pdist2,pred_mu,mvals',repmat({'cityblock'},1,numel(tvals))));
end

%% function for guessing A using existing data

function [A_guess,ss_guess] = eduguess_A(cov_list,mu_list,t_int)
    t1 = randi(numel(mu_list)-2);

    P_1 = cov_list{t1};
    P_2 = cov_list{t1+1};
    Q_1 = cov_list{t1+1};
    Q_2 = cov_list{t1+2};

    Qsqrt = sqrtm(Q_1);
    Psqrt = sqrtm(P_1);

    M = inv(Qsqrt)*Q_2*inv(Qsqrt);
    N = inv(Psqrt)*P_2*inv(Psqrt);

    [U_M,E_M] = eig(M);
    [U_N,E_N] = eig(N);
    [~,idx_M] = sort(diag(E_M));
    [~,idx_N] = sort(diag(E_N));

    U_M = U_M(:,idx_M);
    U_N = U_N(:,idx_N);

    mu_0A = mu_list{t1};
    mu_tA = mu_list{t1+1};
    mu_0B = mu_list{t1+1};
    mu_tB = mu_list{t1+2};

    vec_v = U_M'*inv(Qsqrt)*(mu_tA' - mu_tB');
    vec_w = U_N'*inv(Psqrt)*(mu_0A' - mu_0B');
    Theta = diag(sign(vec_v./vec_w));

    V = Qsqrt*U_M*Theta*U_N'*inv(Psqrt);
    A_guess = real(logm(V)/t_int);

    ss_guess = inv(V-eye(size(V,1)))*(V*mu_0A' - mu_tA' + V*mu_0B' - mu_tB')/2;
end