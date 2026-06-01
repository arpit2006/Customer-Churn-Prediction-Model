import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier,plot_importance
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score

data = pd.read_csv(
    r"D:\Data Science\ML Model\Churn Predictions\Dataset.csv"
).drop("customerID", axis=1)

print(data)

print((data.head()))

print((data.tail()))

print((data.info()))

print((data.describe()))

null_vall = data.isnull().sum()
print("Count of Null Values: ",null_vall)

print(data["Churn"].value_counts())

print(data["TotalCharges"].value_counts())

print(data.dtypes)

data[data["TotalCharges"] == " "]

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"],errors="coerce")

null_cnt = data.isnull().sum()
print("Count of Null Values: ",null_cnt)

data.dropna(inplace=True)

pd.crosstab(
    data["Contract"],
    data["Churn"],
    normalize="index"
) * 100

data.groupby("Churn")["tenure"].describe()

data.groupby("Churn")["MonthlyCharges"].describe()

data["Churn"] = data["Churn"].map({
    "No" : 0,
    "Yes" : 1
})

x = data.drop("Churn",axis = 1)
y = data["Churn"]

print("Input Shape: ",x.shape)
print("Output Shape: ",y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

print("Training Shape: ",x_train.shape)
print("Training Shape: ",y_train.shape)
print("Testing Shape: ",x_test.shape)
print("Testing Shape: ",y_test.shape)

num_attributes = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]
cat_attributes = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

num_pipeline = Pipeline([
    ("Num",SimpleImputer(strategy="mean")),
    ("Normalize",StandardScaler())
])

num_pipeline

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("Encode", OneHotEncoder(handle_unknown="ignore"))
])

cat_pipeline

full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attributes),
    ("cat",cat_pipeline,cat_attributes)
])

full_pipeline

x_train_prepared = full_pipeline.fit_transform(x_train)
x_test_prepared = full_pipeline.transform(x_test)

print("Training Shape: ",x_train_prepared.shape)
print("Testing Shape: ",x_test_prepared.shape)



#Train Model
print("\n===== Logistic Regression Results =====\n")
model = LogisticRegression(n_jobs=1,random_state=42,max_iter=1000)
model.fit(x_train_prepared,y_train)
log_pred = model.predict(x_test_prepared)
accu = accuracy_score(y_test,log_pred)

print("Accuracy Score: ",accu * 100)

print("Precision Score: ",precision_score(y_test,log_pred))

print("Recall Score: ",recall_score(y_test,log_pred))

print("F1 Score: ",f1_score(y_test,log_pred))

print("Error: ", 1 - accu)

print(log_pred[0:10])

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))



#Random Forest Classifier Model
print("\n===== Random Forest Classifier Results =====\n")
ran_model = RandomForestClassifier(n_estimators=42,random_state=42)
ran_model.fit(x_train_prepared,y_train)
ran_pred = ran_model.predict(x_test_prepared)

print("Accuracy:",
      accuracy_score(y_test, ran_pred))

print("Precision Score: ",precision_score(y_test,ran_pred))

print("Recall Score: ",recall_score(y_test,ran_pred))

print("F1 Score: ",f1_score(y_test,ran_pred ))

print("Error: ", 1 - accu)

print("\nClassification Report:")
print(classification_report(y_test, ran_pred ))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, ran_pred ))


print("\nSample Predictions:")
print(ran_pred [:10])



print("\n===== XGB Classifier Results =====\n")

x_model = XGBClassifier(
    n_estimator=200,
     learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

x_model.fit(x_train_prepared,y_train)
x_pred = x_model.predict(x_test_prepared)

print("Accuracy:",
      accuracy_score(y_test, x_pred))

print("Precision Score: ",precision_score(y_test,x_pred))

print("Recall Score: ",recall_score(y_test,x_pred))

print("F1 Score: ",f1_score(y_test,x_pred ))

print("Error: ", 1 - accu)

print("\nClassification Report:")
print(classification_report(y_test, x_pred ))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,x_pred ))


print("\nSample Predictions:")
print(x_pred [:10])

feat_name = full_pipeline.get_feature_names_out()
print("Feature Name: ",feat_name)

importance_df = pd.DataFrame({
    "Feature": feat_name,
    "Importance": x_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
