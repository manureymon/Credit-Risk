%% Load Data
data = readtable('credit_dataset.csv');

% Define the conditions
status0 = strcmp(data.loan_status, 'Fully Paid');
status1 = ismember(data.loan_status, {'Default', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)'});

% Assign the new values
data.loan_status(status0) = {0};
data.loan_status(status1) = {1};

% Remove the rows that are not 0 or 1
valid_rows = status0 | status1;
data = data(valid_rows, :);

% Convert the loan_status column to numeric type
data.loan_status = cell2mat(data.loan_status);

% List of columns to remove
cols_to_remove = {
    'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'acc_now_delinq', ...
    'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m', ...
    'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', ...
    'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', ...
    'total_cu_tl', 'inq_last_12m', 'mths_since_last_major_derog', ...
    'collections_12_mths_ex_med', 'next_pymnt_d', 'collection_recovery_fee', ...
    'id', 'member_id', 'Var1', 'url', 'desc', 'zip_code', 'addr_state', ...
    'mths_since_last_record', 'mths_since_last_delinq', 'out_prncp', ...
    'out_prncp_inv', 'recoveries', 'collection_recovery_fee', 'initial_list_status', 'policy_code', 'total_rec_late_fee',...
    'pub_rec','application_type'};

% Remove columns
cols_to_remove = intersect(cols_to_remove, data.Properties.VariableNames);
data(:, cols_to_remove) = [];

% Convert all columns to double
for i = 1 : width(data)
    % Convert categorical columns to double
    if iscategorical(data{:,i})
        data.(i) = double(data.(i));
    % Convert string, char, cell, datetime columns to double
    elseif isstring(data{:,i}) || ischar(data{:,i}) || iscell(data{:,i}) || isdatetime(data{:,i})
        data.(i) = double(categorical(data.(i))); % First convert to categorical, then to double
    % Convert logical columns directly to double
    elseif islogical(data{:,i})
        data.(i) = double(data.(i));
    end
end

%% Balance data using SMOTE

% Identify good and bad records
good_indices = find(data.loan_status == 0);
bad_indices = find(data.loan_status == 1);

% Display original class distribution
fprintf('Original distribution:\n');
fprintf('Good loans: %d (%.2f%%)\n', length(good_indices), 100*length(good_indices)/height(data));
fprintf('Bad loans: %d (%.2f%%)\n', length(bad_indices), 100*length(bad_indices)/height(data));

% Calculate the imbalance ratio
imbalance_ratio = length(good_indices) / length(bad_indices);
fprintf('Imbalance ratio: %.2f:1\n', imbalance_ratio);

% Separate features and target
X = data;
X.loan_status = [];  % Remove the target variable
y = data.loan_status;

% Create a stratified cross-validation split for training and testing
rng(42); % Set seed for reproducibility
cv = cvpartition(y, 'Holdout', 0.2);
X_train = X(cv.training, :);
y_train = y(cv.training);
X_test = X(cv.test, :);
y_test = y(cv.test);

fprintf('Training set: %d samples\n', height(X_train));
fprintf('Test set: %d samples\n', height(X_test));

% Convert to matrices for SMOTE
X_train_matrix = table2array(X_train);
X_test_matrix = table2array(X_test);

% Apply SMOTE to balance the training data
fprintf('Applying SMOTE to balance the training data...\n');

% Parameters for SMOTE
k = 5; % number of nearest neighbors
target_ratio = 0.5; % ratio of minority to majority samples after SMOTE

% Identify minority class samples
minority_indices = find(y_train == 1);
majority_indices = find(y_train == 0);

% Determine how many synthetic samples to generate
n_minority = length(minority_indices);
n_majority = length(majority_indices);
n_synthetic = round(target_ratio * n_majority) - n_minority;

fprintf('Creating %d synthetic minority samples...\n', n_synthetic);

% Extract minority class samples
minority_samples = X_train_matrix(minority_indices, :);

% Generate synthetic samples
synthetic_samples = zeros(n_synthetic, size(X_train_matrix, 2));
synthetic_labels = ones(n_synthetic, 1);

% For each minority sample
for i = 1:n_synthetic
    % Randomly select a minority sample
    idx = randi(n_minority);
    sample = minority_samples(idx, :);
    
    % Find k nearest neighbors for this sample
    distances = pdist2(sample, minority_samples);
    [~, nn_indices] = sort(distances);
    nn_indices = nn_indices(2:(k+1)); % Skip the first one as it's the sample itself
    
    % Randomly select one of the k neighbors
    nn_idx = nn_indices(randi(length(nn_indices)));
    neighbor = minority_samples(nn_idx, :);
    
    % Generate synthetic sample
    random_weight = rand();
    synthetic_samples(i, :) = sample + random_weight * (neighbor - sample);
end

% Combine original and synthetic data
X_train_balanced_matrix = [X_train_matrix; synthetic_samples];
y_train_balanced = [y_train; synthetic_labels];

% Convert back to table format for creditscorecard function
X_train_balanced = array2table(X_train_balanced_matrix, 'VariableNames', X_train.Properties.VariableNames);
X_train_balanced.loan_status = y_train_balanced;

% Display balanced distribution
fprintf('Balanced training distribution:\n');
fprintf('Good loans: %d (%.2f%%)\n', sum(y_train_balanced == 0), 100*sum(y_train_balanced == 0)/length(y_train_balanced));
fprintf('Bad loans: %d (%.2f%%)\n', sum(y_train_balanced == 1), 100*sum(y_train_balanced == 1)/length(y_train_balanced));


%% Credit Scorecard
predictors = setdiff(X_train_balanced.Properties.VariableNames, {'loan_status'});

% Train the initial scorecard
sc = creditscorecard(X_train_balanced, ...
    'IDVar', '', ...
    'ResponseVar', 'loan_status', ...
    'GoodLabel', 0, ...
    'PredictorVars', predictors);

sc = autobinning(sc);
sc = fitmodel(sc);
sc_formatted = formatpoints(sc, 'PointsOddsAndPDO', [600 2 20], 'Round', 'AllPoints');

sc_points = displaypoints(sc_formatted);

% Calculate credit scores
[scores, ~] = score(sc_formatted, data(:, predictors));

% Add scores and probabilities to the original dataset
data.credit_score = scores;

%% Score Distribution Visualization

figure;
hold on;

% Histogram for good loans (status=0) in green
histogram(scores(data.loan_status == 0), 'FaceColor', 'green', 'FaceAlpha', 0.6, 'BinWidth', 20);

% Histogram for bad loans (status=1) in red
histogram(scores(data.loan_status == 1), 'FaceColor', 'red', 'FaceAlpha', 0.6, 'BinWidth', 20);

% Add the threshold line at 600
line([580 580], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);

title('Credit Score Distribution by Loan Type');
xlabel('Credit Score');
ylabel('Frequency');
legend('Good Loans (Status=0)', 'Bad Loans (Status=1)');
hold off;

%% Performance Metrics
% Create binary prediction where scores >= 580 are predicted as good loans (0) and scores < 580 are predicted as bad loans (1)
y_pred_score = double(scores < 580);

% The actual loan status is already in data.loan_status (0 for good, 1 for bad)
y_true = data.loan_status;

% Calculate accuracy
accuracy = sum(y_pred_score == y_true) / length(y_true);
fprintf('Accuracy: %.4f (%.2f%%)\n', accuracy, accuracy * 100);

% Create confusion matrix
cm = confusionmat(y_true, y_pred_score);

% Display confusion matrix
figure;
confusionchart(cm, {'Good Loan (0)', 'Bad Loan (1)'});
title('Confusion Matrix: True vs Predicted Loan Status');

% Calculate performance metrics
TP = cm(2,2);  % True Positives (correctly predicted bad loans)
FP = cm(1,2);  % False Positives (good loans predicted as bad)
TN = cm(1,1);  % True Negatives (correctly predicted good loans)
FN = cm(2,1);  % False Negatives (bad loans predicted as good)

precision = TP / (TP + FP);
recall = TP / (TP + FN);  % Also known as sensitivity or TPR
specificity = TN / (TN + FP);
f1_score = 2 * (precision * recall) / (precision + recall);

% Display metrics
fprintf('\nModel Performance Metrics:\n');
fprintf('Precision: %.4f\n', precision);
fprintf('Recall/Sensitivity: %.4f\n', recall);
fprintf('Specificity: %.4f\n', specificity);
fprintf('F1 Score: %.4f\n', f1_score);