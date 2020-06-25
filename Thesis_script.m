%% All data
clear all
data = readtable('DATA.xlsx', 'PreserveVariableNames', true);
dates = table2array(data(:,1)); % ECB announcement dates
de_1y = table2array(data(:,4)); % Deutsche Bund (1 YTM)
de_2y = table2array(data(:,5)); % Deutsche Bund (2 YTM)
eurpln = table2array(data(:,7)); 
differential = table2array(data(:,13)); % Interest rate differential
ois_1w = table2array(data(:,14)); % Overnight Interest Swap (1 week)
ois_1m = table2array(data(:,15)); 
wibor_change = table2array(data(:,16)); 
euribor_change = table2array(data(:,17));
carry_returns_alt = table2array(data(:,18));

%% Create plots

figure(1)
plot(dates(1:174), euribor_change(1:174));
xlabel('Time');
title('Changes in the EURIBOR');

figure(2)
plot(dates(1:174), wibor_change(1:174));
xlabel('Time');
title('Changes in the WIBOR');

figure(4)
plot(dates(1:174), de_1y(1:174))
xlabel('Time');
title('Shocks in Deutsche Bund (1 YTM)');

figure(5)
plot(dates(1:174), de_2y(1:174))
xlabel('Time');
title('Shocks in Deutsche Bund (2 YTM)');

figure(6)
plot(dates(1:174), ois_1m(1:174))
xlabel('Time');
title('Shocks in OIS (1 month to maturity)');

figure(7) 
plot(dates(1:174), ois_1w(1:174))
xlabel('Time');
title('Shocks in OIS (1 week to maturity)');

figure(8)
plot(dates(1:174), eurpln(1:174))
xlabel('Time');
title('Exchange rate');

figure(9)
plot(dates(1:174), carry_returns_alt(1:174));
xlabel('Time');
title('Annualized returns in percentage points');

%% Testing UIP
X_bar = mean(carry_returns_alt);
sigma = std(carry_returns_alt);
n_obs = 174;
mu = 0;

CI = [X_bar - norminv(0.975)*sigma/sqrt(n_obs), X_bar + norminv(0.975)*sigma/sqrt(n_obs)]
z_statistic = sqrt(n_obs)*(X_bar - mu)/sigma
p_value = 2*(1 - normcdf(z_statistic))

%% Data selection

% Deutsche Bunds: press conference window
% Overnight Interest Swap: press release window
% Meetings once per month: data points 18 - 133
% Meetings every six weeks: data points 134 - 174
% Before data point 18, Wibor rates are highly volatile

period = 134:174;
nLags = 4; 
L_period = period - 1;

%% Predictability of carry returns using returns on previous trades

% Regression of returns on the one week post-GC carry trade 
% on returns generated around previous carry trades

y = carry_returns_alt(period);
X = [ones(length(period),1) carry_returns_alt(L_period)];
for i = 0:nLags-1
    X(:,i+2) = carry_returns_alt(L_period - i);
end

beta_hat=inv(X'*X)*(X'*y);       % Computation of coefficients
py = X*beta_hat;                 % Fitted values
u = y - py;                      % Error terms

% Creating robust standard errors
G = [repmat(u, [1 size(X,2)]) ].*X;
vwhite = inv(X'*X)*(G'*G)*inv(X'*X);
robust_standard_errors=sqrt(diag(vwhite));

% t-statistic
t_statistic = abs(beta_hat./robust_standard_errors);
p_values = (1 - normcdf(t_statistic))*2;

% Confidence intervals
CI_coefficients = [beta_hat - norminv(0.9750)*robust_standard_errors, beta_hat + norminv(0.9750)*robust_standard_errors];

% Regression output
A = [beta_hat CI_coefficients robust_standard_errors t_statistic p_values]

% Wald statistics for joint significance of lags
range = 2:length(beta_hat);
beta = beta_hat(range);             % beta contains the estimates that are being tested
V = vwhite(range, range);           % matrix V 

% Create the Wald test statistic
% In this piece of the code, the Wald statistic is computed in order to 
% test the joint significance of the coefficients:
% - R denotes the following matrix, in this particular case: 
%   R = [1,0;0,1] 
% - q denotes the restricted values, i.e. q = [0 0]'
% - beta consists of the vector of coefficients that are being tested
% V contains the variance-covariance matrix of the estimated coefficients
Wald_1 = beta';                     % Wald_1 = R*beta - q 
Wald_2 = V;                         % Wald_2 = R*V*R' 
Wald_3 = Wald_1';                   % Wald_3 = (R*beta - q)'
n = length(beta);                   % number of restrictions 

statistic = Wald_1*inv(Wald_2)*Wald_3
critical_value = chi2inv(0.95,n)
p_value_model = chi2cdf(statistic, n, 'upper')


%% Predictability of carry returns using shocks around GC-announcements
%% OIS_1W
% Regression of returns on the one week post-GC carry trade 
% on returns generated around shocks around GC-announcements

y = carry_returns_alt(period);
X = [ones(length(period),1) ois_1w(period)];
for i = 0:nLags-1
    X(:,i+3) = ois_1w(L_period - i);
end

beta_hat=inv(X'*X)*(X'*y);       % Computation of coefficients
py = X*beta_hat;                 % Fitted values
u = y - py;                      % Error terms

% Creating robust standard errors
G = [repmat(u, [1 size(X,2)]) ].*X;
vwhite = inv(X'*X)*(G'*G)*inv(X'*X);
robust_standard_errors=sqrt(diag(vwhite));

% t-statistic
t_statistic = abs(beta_hat./robust_standard_errors);
p_values = (1 - normcdf(t_statistic))*2;

% Confidence intervals
CI_coefficients = [beta_hat - norminv(0.9750)*robust_standard_errors, beta_hat + norminv(0.9750)*robust_standard_errors];

% Regression output
A = [beta_hat CI_coefficients robust_standard_errors t_statistic p_values]

% Wald - statistics 
range = 3:length(beta_hat);
beta = beta_hat(range);             % beta contains the estimates that are being tested
V = vwhite(range, range);           % matrix V 

% Create the Wald test statistic
% In this piece of the code, the Wald statistic is computed in order to 
% test the joint significance of the coefficients:
% - R denotes the following matrix, in this particular case: 
%   R = [1,0;0,1] 
% - q denotes the restricted values, i.e. q = [0 0]'
% - beta consists of the vector of coefficients that are being tested
% V contains the variance-covariance matrix of the estimated coefficients
Wald_1 = beta';                     % Wald_1 = R*beta - q 
Wald_2 = V;                         % Wald_2 = R*V*R' 
Wald_3 = Wald_1';                   % Wald_3 = (R*beta - q)'
n = length(beta);                   % number of restrictions 

statistic = Wald_1*inv(Wald_2)*Wald_3
critical_value = chi2inv(0.95,n)
p_value_model = chi2cdf(statistic, n, 'upper')

%% OIS_1M
% Regression of returns on the one week post-GC carry trade 
% on returns generated around shocks around GC-announcements

y = carry_returns_alt(period);
X = [ones(length(period),1) ois_1m(period)];
for i = 0:nLags-1
    X(:,i+3) = ois_1m(L_period - i);
end

beta_hat=inv(X'*X)*(X'*y);       % Computation of coefficients
py = X*beta_hat;                 % Fitted values
u = y - py;                      % Error terms

% Creating robust standard errors
G = [repmat(u, [1 size(X,2)]) ].*X;
vwhite = inv(X'*X)*(G'*G)*inv(X'*X);
robust_standard_errors=sqrt(diag(vwhite));

% t-statistic
t_statistic = abs(beta_hat./robust_standard_errors);
p_values = (1 - normcdf(t_statistic))*2;

% Confidence intervals
CI_coefficients = [beta_hat - norminv(0.9750)*robust_standard_errors, beta_hat + norminv(0.9750)*robust_standard_errors];

% Regression output
A = [beta_hat CI_coefficients robust_standard_errors t_statistic p_values]

% Wald - statistics 
range = 3:length(beta_hat);
beta = beta_hat(range);             % beta contains the estimates that are being tested
V = vwhite(range, range);           % matrix V 

% Create the Wald test statistic
% In this piece of the code, the Wald statistic is computed in order to 
% test the joint significance of the coefficients:
% - R denotes the following matrix, in this particular case: 
%   R = [1,0;0,1] 
% - q denotes the restricted values, i.e. q = [0 0]'
% - beta consists of the vector of coefficients that are being tested
% V contains the variance-covariance matrix of the estimated coefficients
Wald_1 = beta';                     % Wald_1 = R*beta - q 
Wald_2 = V;                         % Wald_2 = R*V*R' 
Wald_3 = Wald_1';                   % Wald_3 = (R*beta - q)'
n = length(beta);                   % number of restrictions 

statistic = Wald_1*inv(Wald_2)*Wald_3
critical_value = chi2inv(0.95,n)
p_value_model = chi2cdf(statistic, n, 'upper')

%% DE_1Y
% Regression of returns on the one week post-GC carry trade 
% on returns generated around shocks around GC-announcements

y = carry_returns_alt(period);
X = [ones(length(period),1) de_1y(period)];
for i = 0:nLags-1
    X(:,i+3) = de_1y(L_period - i);
end

beta_hat=inv(X'*X)*(X'*y);       % Computation of coefficients
py = X*beta_hat;                 % Fitted values
u = y - py;                      % Error terms

% Creating robust standard errors
G = [repmat(u, [1 size(X,2)]) ].*X;
vwhite = inv(X'*X)*(G'*G)*inv(X'*X);
robust_standard_errors=sqrt(diag(vwhite));

% t-statistic
t_statistic = abs(beta_hat./robust_standard_errors);
p_values = (1 - normcdf(t_statistic))*2;

% Confidence intervals
CI_coefficients = [beta_hat - norminv(0.9750)*robust_standard_errors, beta_hat + norminv(0.9750)*robust_standard_errors];

% Regression output
A = [beta_hat CI_coefficients robust_standard_errors t_statistic p_values]

% Wald - statistics 
range = 3:length(beta_hat);
beta = beta_hat(range);             % beta contains the estimates that are being tested
V = vwhite(range, range);           % matrix V 

% Create the Wald test statistic
% In this piece of the code, the Wald statistic is computed in order to 
% test the joint significance of the coefficients:
% - R denotes the following matrix, in this particular case: 
%   R = [1,0;0,1] 
% - q denotes the restricted values, i.e. q = [0 0]'
% - beta consists of the vector of coefficients that are being tested
% V contains the variance-covariance matrix of the estimated coefficients
Wald_1 = beta';                     % Wald_1 = R*beta - q 
Wald_2 = V;                         % Wald_2 = R*V*R' 
Wald_3 = Wald_1';                   % Wald_3 = (R*beta - q)'
n = length(beta);                   % number of restrictions 

statistic = Wald_1*inv(Wald_2)*Wald_3
critical_value = chi2inv(0.95,n)
p_value_model = chi2cdf(statistic, n, 'upper')

%% DE_2Y
% Regression of returns on the one week post-GC carry trade 
% on returns generated around shocks around GC-announcements

y = carry_returns_alt(period);
X = [ones(length(period),1) de_2y(period)];
for i = 0:nLags-1
    X(:,i+3) = de_2y(L_period - i);
end

beta_hat=inv(X'*X)*(X'*y);       % Computation of coefficients
py = X*beta_hat;                 % Fitted values
u = y - py;                      % Error terms

% Creating robust standard errors
G = [repmat(u, [1 size(X,2)]) ].*X;
vwhite = inv(X'*X)*(G'*G)*inv(X'*X);
robust_standard_errors=sqrt(diag(vwhite));

% t-statistic
t_statistic = abs(beta_hat./robust_standard_errors);
p_values = (1 - normcdf(t_statistic))*2;

% Confidence intervals
CI_coefficients = [beta_hat - norminv(0.9750)*robust_standard_errors, beta_hat + norminv(0.9750)*robust_standard_errors];

% Regression output
A = [beta_hat CI_coefficients robust_standard_errors t_statistic p_values]

% Wald - statistics 
range = 3:length(beta_hat);
beta = beta_hat(range);             % beta contains the estimates that are being tested
V = vwhite(range, range);           % matrix V 

% Create the Wald test statistic
% In this piece of the code, the Wald statistic is computed in order to 
% test the joint significance of the coefficients:
% - R denotes the following matrix, in this particular case: 
%   R = [1,0;0,1] 
% - q denotes the restricted values, i.e. q = [0 0]'
% - beta consists of the vector of coefficients that are being tested
% V contains the variance-covariance matrix of the estimated coefficients
Wald_1 = beta';                     % Wald_1 = R*beta - q 
Wald_2 = V;                         % Wald_2 = R*V*R' 
Wald_3 = Wald_1';                   % Wald_3 = (R*beta - q)'
n = length(beta);                   % number of restrictions 

statistic = Wald_1*inv(Wald_2)*Wald_3
critical_value = chi2inv(0.95,n)
p_value_model = chi2cdf(statistic, n, 'upper')
%% Autoregressive Distributed Lag model
%% Lagged returns + OIS_1M

y = carry_returns_alt(period);
X = [ones(length(period),1) carry_returns_alt(L_period) ois_1m(period)];
for i = 0:nLags-1
    X(:,i+4) = ois_1m(L_period - i);
end

beta_hat=inv(X'*X)*(X'*y);       % Computation of coefficients
py = X*beta_hat;                 % Fitted values
u = y - py;                      % Error terms

% Creating robust standard errors
G = [repmat(u, [1 size(X,2)]) ].*X;
vwhite = inv(X'*X)*(G'*G)*inv(X'*X);
robust_standard_errors=sqrt(diag(vwhite));

% t-statistic
t_statistic = abs(beta_hat./robust_standard_errors);
p_values = (1 - normcdf(t_statistic))*2;

% Confidence intervals
CI_coefficients = [beta_hat - norminv(0.9750)*robust_standard_errors, beta_hat + norminv(0.9750)*robust_standard_errors];

% Regression output
A = [beta_hat CI_coefficients robust_standard_errors t_statistic p_values]

% Wald - statistics
range = 2:length(beta_hat);
beta = beta_hat(range);             % beta contains the estimates that are being tested
V = vwhite(range, range);           % matrix V 

% Create the Wald test statistic
% In this piece of the code, the Wald statistic is computed in order to 
% test the joint significance of the coefficients:
% - R denotes the following matrix, in this particular case: 
%   R = [1,0;0,1] 
% - q denotes the restricted values, i.e. q = [0 0]'
% - beta consists of the vector of coefficients that are being tested
% V contains the variance-covariance matrix of the estimated coefficients
Wald_1 = beta';                     % Wald_1 = R*beta - q 
Wald_2 = V;                         % Wald_2 = R*V*R' 
Wald_3 = Wald_1';                   % Wald_3 = (R*beta - q)'
n = length(beta);                   % number of restrictions 

statistic = Wald_1*inv(Wald_2)*Wald_3
critical_value = chi2inv(0.95,n)
p_value_model = chi2cdf(statistic, n, 'upper')

%% Dynamic multipliers for ADL-model
% Initialization
dm = zeros(24,1);
dm(1) = beta_hat(3);    
% Horizon < maximum lag length
for i = 2:nLags+1
    dm(i) = beta_hat(i + 2) + beta_hat(2)*dm(i - 1); 
end
% Horizon > maximum lag length
for i = (nLags + 2):24
    dm(i) = dm(i-1)*beta_hat(2);
end
% Create plots
k = 1:24;
plot(k, dm);
xlabel('Time horizon k');
ylabel('Values');
title('Dynamic multipliers');
xlim([1 24]);

%% Lagged returns + OIS_1W

y = carry_returns_alt(period);
X = [ones(length(period),1) carry_returns_alt(L_period) ois_1w(period)];
for i = 0:nLags-1
    X(:,i+4) = ois_1w(L_period - i);
end

beta_hat=inv(X'*X)*(X'*y);       % Computation of coefficients
py = X*beta_hat;                 % Fitted values
u = y - py;                      % Error terms

% Creating robust standard errors
G = [repmat(u, [1 size(X,2)]) ].*X;
vwhite = inv(X'*X)*(G'*G)*inv(X'*X);
robust_standard_errors=sqrt(diag(vwhite));

% t-statistic
t_statistic = abs(beta_hat./robust_standard_errors);
p_values = (1 - normcdf(t_statistic))*2;

% Confidence intervals
CI_coefficients = [beta_hat - norminv(0.9750)*robust_standard_errors, beta_hat + norminv(0.9750)*robust_standard_errors];

% Regression output
A = [beta_hat CI_coefficients robust_standard_errors t_statistic p_values]

% Wald - statistics
range = 2:length(beta_hat);
beta = beta_hat(range);             % beta contains the estimates that are being tested
V = vwhite(range, range);           % matrix V 

% Create the Wald test statistic
% In this piece of the code, the Wald statistic is computed in order to 
% test the joint significance of the coefficients:
% - R denotes the following matrix, in this particular case: 
%   R = [1,0;0,1] 
% - q denotes the restricted values, i.e. q = [0 0]'
% - beta consists of the vector of coefficients that are being tested
% V contains the variance-covariance matrix of the estimated coefficients
Wald_1 = beta';                     % Wald_1 = R*beta - q 
Wald_2 = V;                         % Wald_2 = R*V*R' 
Wald_3 = Wald_1';                   % Wald_3 = (R*beta - q)'
n = length(beta);                   % number of restrictions 

statistic = Wald_1*inv(Wald_2)*Wald_3
critical_value = chi2inv(0.95,n)
p_value_model = chi2cdf(statistic, n, 'upper')


%% Dynamic multipliers for ADL-model
% Initialization
dm = zeros(24,1);
dm(1) = beta_hat(3);    
% Horizon < maximum lag length
for i = 2:nLags+1
    dm(i) = beta_hat(i + 2) + beta_hat(2)*dm(i - 1); 
end
% Horizon > maximum lag length
for i = (nLags + 2):24
    dm(i) = dm(i-1)*beta_hat(2);
end
% Create plots
k = 1:24;
plot(k, dm);
xlabel('Time horizon k');
ylabel('Values');
title('Dynamic multipliers');
xlim([1 24]);





