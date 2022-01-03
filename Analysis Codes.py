import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


dataset_root = 'NSL-KDD-Dataset/NSL-KDD-Dataset'

train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')


# Original KDD dataset feature names obtained from 
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']


# Differentiating between nominal, binary, and numeric features

# root_shell is marked as a continuous feature in the kddcup.names 
# file, but it is supposed to be a binary feature according to the 
# dataset documentation

col_names = np.array(header_names)

nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]
numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))

nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()

# training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories
# file obtained from http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types



category = defaultdict(list)
category['benign'].append('normal')

with open('NSL-KDD-Dataset/NSL-KDD-Dataset/training_attack_types.txt', 'r') as f:
    for line in f.readlines():
        attack, cat = line.strip().split(' ')
        category[cat].append(attack)

attack_mapping = dict((v,k) for k in category for v in category[k])


train_df = pd.read_csv(train_file, names=header_names)
train_df['attack_category'] = train_df['attack_type'] \
                                .map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)
    
test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'] \
                                .map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)


train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()

test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)

train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)

test_attack_types.plot(kind='barh', figsize=(20,10), fontsize=15)

test_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)

# Let's take a look at the binary features
# By definition, all of these features should have a min of 0.0 and a max of 1.0
#execute the commands in console

train_df[binary_cols].describe().transpose()


# Wait a minute... the su_attempted column has a max value of 2.0?

train_df.groupby(['su_attempted']).size()

# Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0

train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)
train_df.groupby(['su_attempted']).size()


# Next, we notice that the num_outbound_cmds column only takes on one value!

train_df.groupby(['num_outbound_cmds']).size()

# Now, that's not a very useful feature - let's drop it from the dataset

train_df.drop('num_outbound_cmds', axis = 1, inplace=True)
test_df.drop('num_outbound_cmds', axis = 1, inplace=True)
numeric_cols.remove('num_outbound_cmds')


"""
Data Preparation

"""
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category','attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category','attack_type'], axis=1)




combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# Store dummy variable feature names
dummy_variables = list(set(train_x)-set(combined_df_raw))

#execute the commands in console
train_x.describe()
train_x['duration'].describe()
# Experimenting with StandardScaler on the single 'duration' feature
from sklearn.preprocessing import StandardScaler

durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)
pd.Series(scaled_durations.flatten()).describe()

# Experimenting with MinMaxScaler on the single 'duration' feature
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
pd.Series(min_max_scaled_durations.flatten()).describe()

# Experimenting with RobustScaler on the single 'duration' feature
from sklearn.preprocessing import RobustScaler

min_max_scaler = RobustScaler().fit(durations)
robust_scaled_durations = min_max_scaler.transform(durations)
pd.Series(robust_scaled_durations.flatten()).describe()

# Let's proceed with StandardScaler- Apply to all the numeric columns

standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = \
    standard_scaler.transform(train_x[numeric_cols])

test_x[numeric_cols] = \
    standard_scaler.transform(test_x[numeric_cols])
    
train_x.describe()


train_Y_bin = train_Y.apply(lambda x: 0 if x is 'benign' else 1)
test_Y_bin = test_Y.apply(lambda x: 0 if x is 'benign' else 1)


#Step 4: Data exporation
list(train_x.columns.values)    #check the column names
print(len(test_x))              #check the length of the testing set 
print(len(train_x))             #check the length of the traning set 
print(train_attack_cats)        #check the sample for each training class

# 5-class classification version
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import classification_report
import time  #time logs


#Step 6: Predictive Modeling

#Decision Tree Classifier
print('******************Decision Tree Classifier************************')
start_time = time.time()
DTC = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
        max_features=None, max_leaf_nodes=1000, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        presort=False, random_state=15, splitter='best')
DTC_pred_y = DTC.fit(train_x, train_Y).predict(test_x)
DTC_results = confusion_matrix(test_Y, DTC_pred_y)
DTC_error = zero_one_loss(test_Y, DTC_pred_y)
print('Matrix Result:\n', DTC_results)
print('Error Rate: ', DTC_error)
print (classification_report(test_Y, DTC_pred_y,digits=4))
print("--- %s seconds ---" % (time.time() - start_time))

#MLP Classifier
print('******************Multi-Layer Perceptron Classifier************************')
start_time = time.time()
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=100, activation ='logistic',
                    solver = 'sgd')
#, alpha=0.0001, batch_size = 1, learning_rate_init=0.001, power_t=0.5, max_iter=200, random_state=15, momentum = 0.9, max_fun=15000
MLP_pred_y = MLP.fit(train_x, train_Y).predict(test_x)
MLP_results = confusion_matrix(test_Y, MLP_pred_y)
MLP_error = zero_one_loss(test_Y, MLP_pred_y)
print('Matrix Result:\n', MLP_results)
print('Error Rate: ', MLP_error)
print (classification_report(test_Y, MLP_pred_y,digits=4))
print("--- %s seconds ---" % (time.time() - start_time))


#RandomForest Classifier
print('******************RandomForest Classifier************************')
start_time = time.time()
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                             min_impurity_split=1e-07, min_samples_leaf=1,
                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                             n_estimators=10, n_jobs=1, oob_score=False, 
                             random_state=45, verbose=0, warm_start=False)
RFC_pred_y = RFC.fit(train_x, train_Y).predict(test_x)
RFC_results = confusion_matrix(test_Y, RFC_pred_y)
RFC_error = zero_one_loss(test_Y, RFC_pred_y)
print('Matrix Result:\n', RFC_results)
print('Error Rate: ', RFC_error)
print (classification_report(test_Y, RFC_pred_y,digits=4))
print("--- %s seconds ---" % (time.time() - start_time))



#Support Vector Classifier
print('******************Support Vector Classifier************************')
start_time = time.time()
from sklearn.svm import SVC
clf = SVC(kernel= 'linear', gamma = 'scale', C = 0.5)
SVC_pred_y = clf.fit(train_x, train_Y).predict(test_x)
SVC_results = confusion_matrix(test_Y, SVC_pred_y)
SVC_error = zero_one_loss(test_Y, SVC_pred_y)
print('Matrix Result:\n', SVC_results)
print('Error Rate: ', SVC_error)
print (classification_report(test_Y, SVC_pred_y,digits=4))
print("--- %s seconds ---" % (time.time() - start_time))



#Gaussian Naive Bayes
print('******************Naive Bayes Classifier************************')
start_time = time.time()
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB(var_smoothing=0.05)
NB_pred_y = NB.fit(train_x, train_Y).predict(test_x)
NB_results = confusion_matrix(test_Y, NB_pred_y)
NB_error = zero_one_loss(test_Y, NB_pred_y)
print('Matrix Result:\n', NB_results)
print('Error Rate: ', NB_error)
print (classification_report(test_Y, NB_pred_y,digits=4))
print("--- %s seconds ---" % (time.time() - start_time))


#Step 7: Plot the accuracy of each model
plt.clf()   # clear existing plt plot
plt.ylim(0.4,1)
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.bar(['DTC','MLP','RFC','SVC','NB'], 
        [1-DTC_error,1-MLP_error,1-RFC_error,1-SVC_error,1-NB_error],
        color='blue')
plt.show()

#Plot confusion matrix for each scenario
from sklearn.metrics import plot_confusion_matrix
DTC_plot = plot_confusion_matrix(DTC,test_x,test_Y, cmap=plt.cm.Reds,normalize='true')
DTC_plot.ax_.set_title('Decision Tree')
MLP_plot = plot_confusion_matrix(MLP,test_x,test_Y, cmap=plt.cm.Reds,normalize='true')
MLP_plot.ax_.set_title('Multi-Layer Perceptron')
RFC_plot = plot_confusion_matrix(RFC,test_x,test_Y, cmap=plt.cm.Reds,normalize='true')
RFC_plot.ax_.set_title('Random Forest')
SVC_plot = plot_confusion_matrix(clf,test_x,test_Y,cmap=plt.cm.Reds, normalize='true')
SVC_plot.ax_.set_title('Support Vector')
NB_plot = plot_confusion_matrix(NB,test_x,test_Y,cmap=plt.cm.Reds, normalize='true')
NB_plot.ax_.set_title('Naive Bayes')
