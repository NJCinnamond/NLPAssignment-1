import pandas as pd 

df1 = pd.read_csv("perceptron_propername_test_predictions.csv")
df2 = pd.read_csv("perceptron_propername_test_predictions_mask_devacc_82.7.csv")
df3 = pd.read_csv("perceptron_propername_test_predictions_nomask_devacc80.6.csv")

print(df1.equals(df2))
print(df1.equals(df3))
