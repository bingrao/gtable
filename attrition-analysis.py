import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import operator

warnings.simplefilter('ignore')


def load_attrition_data():
    dataset = pd.read_csv("data/employee-attrition.csv")
    return dataset


def pdistcompute(attrition, cols):
    # attrition is the dataframe
    # cols is the subset of columns
    attrition = attrition[cols]
    pair_wise = pd.Series(pdist(attrition, 'cosine'))  # finding pairwise distance between data
    count = pair_wise.groupby(
        pd.cut(pair_wise, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])).count()  # grouping based on values
    # plotting
    plt.plot(np.arange(len(count)), count)  # general plot for all users.
    return count


attrition = load_attrition_data()
print(attrition.columns.values)
print(attrition.shape)
attrition.head()

attrition_encoded = attrition.iloc[:, 1:].apply(LabelEncoder().fit_transform)  # encoding on all except Age
attrition = pd.concat([attrition.iloc[:, 0], attrition_encoded], axis=1, sort=False)
attrition.head()

full_cols = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
             'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
             'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
             'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
             'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
             'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
             'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
             'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
             'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
             'YearsWithCurrManager']


def privacy_apriori_analysis(full_cols):
    print("<=0.5 implies COMPLETE PRIVACY\n>0.5 implies PRIVACY VIOLATING ATTRIBUTE")
    fset_80 = []  # With value greater than 0.80
    fset_50 = []  # with value greater than 0.5 but less than 80
    fset_low = []
    for i in range(len(full_cols)):
        cols = full_cols[i]
        cols = [cols] + ['YearsWithCurrManager']
        count = pdistcompute(attrition, cols)
        if (full_cols[i] != 'YearsWithCurrManager') & (np.sum(count) != 0):
            # YearsWithCurrManager is used as reference and ignored for analysis
            # count = 0 implies all same values for col
            # print(full_cols[i] + str(":\t") + str(count[0]/sum(count)))
            if count[0] / sum(count) >= 0.8:
                fset_80.append(full_cols[i])
            if (count[0] / sum(count) < 0.8) & (count[0] / sum(count) >= 0.5):
                fset_50.append(full_cols[i])
            if count[0] / sum(count) < 0.5:
                fset_low.append(full_cols[i])
    return fset_80, fset_50, fset_low


private_attr = []  # Contains all list of private attributes
fset_80, fset_50, fset_low = privacy_apriori_analysis(full_cols)
fset_80  # first set of privacy violating attributes
private_attr = fset_80

fset_80

fset_50  # second set of quasi identifiers

fset_low  # Lower risk variables


def privacy_attr_apriori_2(attrition, fset_50, fset):
    # fset -> fset_50 or fset_low
    second_list = [];
    for i in range(len(fset_50)):
        for j in range(len(fset)):
            if fset_50[i] != fset[j]:
                cols = [fset_50[i]] + [fset[j]]
                count = pdistcompute(attrition, cols)
                # print(set(cols))
                if count[0] / sum(count) > 0.75:
                    # print(cols, str(count[0]/sum(count)))
                    second_list.append(cols[1])
    return (second_list)


# CHANGE THIS FUNCTION TO PRINT TOP FIVE
def most_common(lst):
    return max(set(lst), key=lst.count)


# Removing Quasi identifiers
second_list = privacy_attr_apriori_2(attrition, fset_50, fset_50)
print(most_common(second_list))
private_attr.append(most_common(second_list))  # Contains all list of private attributes
fset_50.remove(most_common(second_list))

# second_list = privacy_attr_apriori_2(attrition, fset_50, fset_50)
# print(most_common(second_list))
# private_attr.append(most_common(second_list))  # Contains all list of private attributes
# fset_50.remove(most_common(second_list))

third_list = privacy_attr_apriori_2(attrition, fset_50, fset_low)
print(most_common(third_list))
private_attr.append(most_common(third_list))

private_attr  # First metric Using frequent mining


# Second Metric - unique_attr
def unique_feat(attrition, cols):
    attrition = attrition[cols]
    return len(np.unique(attrition))


ulst = {}  # dictionary containing col name and values
for i in range(len(full_cols)):
    cols = full_cols[i]
    ulst[cols] = len(attrition) / unique_feat(attrition, cols)

sorted_ulst = sorted(ulst.items(), key=operator.itemgetter(1))
# print("Attributes, Average Group Size")
unique_attr = []
for k, v in sorted_ulst:
    if v < 55:
        # print(k,v)
        unique_attr.append(k)

unique_attr

# Uniqueness fails to look for uniqueness within an attribute
# Third Metric - Finding uniqueness within an attribute - in imbalanced dataset
df = attrition.groupby('Age')['Age'].count()
print(df.min())

alst = {}
for i in range(len(full_cols)):
    cols = full_cols[i]
    mval = (attrition.groupby(cols)[cols].count()).min()
    print(cols + str(':') + str(mval))
    alst[cols] = 1 / mval

sorted_alst = sorted(alst.items(), key=operator.itemgetter(1))
# print("Attributes, Average Group Size")
imbalance_attr = []
threshold = 0.2  # (1 in 20 records)
for k, v in sorted_alst:
    if v > threshold:
        # print(k,v)
        imbalance_attr.append(k)
imbalance_attr


clst = {}
threshold = 0.95  # (1 in 10 records)
for i in range(len(full_cols)):
    for j in range(len(full_cols)):
        if full_cols[i] not in imbalance_attr:
            if full_cols[j] not in imbalance_attr:
                if full_cols[i] != full_cols[j]:
                    cols = [full_cols[i]] + [full_cols[j]]
                    mval = (attrition.groupby(cols)[cols].count()).min()
                    value = 1 / mval[0]
                    if value > threshold:
                        # print(str(cols) + str(value))
                        if cols[0] not in clst.keys():
                            clst[cols[0]] = 1
                        else:
                            clst[cols[0]] = clst[cols[0]] + 1
sorted_clst = sorted(clst.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_clst)
second_attr = []
count = 0
for k, v in sorted_clst:
    if count < 5:
        second_attr.append(k)
        count = count + 1;

    else:
        break;
second_attr

# # Transformation Logic
# #Step1: Find features that can lead to better prediction - f_subset: subset of features used for task prediction 
# #Step2: pdistcompute on dataframe(f_subset) to find unique ones that can be used to distinguish users 
# #@ADVERSARY: semi-honest adversary who uses all insider knowledge to learn aout user private information; 
# #@ADVERSARY: One who is knowledgable about data preparation  
# 
# #Objective 1: Protect identified sensitive attributes (Age,Distance) so @ADVERSARY cannot de-identify individual 
# #These attributes are ones that can be used by adversary to identify individuals using age, gender, location (PUBLIC). 
# #Using DE-IDENTIFICATION, PRIVATE information such as monthly income, monthly rate, daily rate, percent salary hike, performance rating etc...
# #Protect deidentification using PUBLIC attributes which will protect PRIVATE attributes 
# 
# #Objective 2: Protect sensitive hidden inferences from published data - a case where same data can be used
# #to make multiple classes - using attrition data to predict suicide 
#


from sklearn.model_selection import train_test_split

# Step 1 using a classifier to predict attrition from input data
feat = ['Age', 'BusinessTravel', 'DailyRate', 'Department',
        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager']

label = ['Attrition']
X = attrition[feat]
y = attrition[label]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80% training and 30% test

PRIVACY_FLAG = 1
if PRIVACY_FLAG == 1:
    for ele in private_attr:
        feat.remove(ele)
        X_train = X_train[feat]
        X_test = X_test[feat]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# In[187]:


# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)
# Train the app using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
plt.barh(feat, clf.feature_importances_)
plt.yticks(fontsize=7)
plt.tight_layout()
