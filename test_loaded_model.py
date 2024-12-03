import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

df_pred = pd.read_csv("CSCI-443/preds_30fps_right.csv")

loaded_model = joblib.load("CSCI-443/traffic_prediction_model.pkl")

# label encode the Day of the week column
le = LabelEncoder()
df_pred['Day of the week'] = le.fit_transform(df_pred['Day of the week'])

preds = loaded_model.predict(df_pred)
# preds = le.inverse_transform(preds)
# Label encoded values mapped to:
# 0: heavy
# 1: high
# 2: low
# 3: normal
labels = ['heavy' 'high' 'low' 'normal']
print(set(preds))
# print each prediction in correlation to the label
for pred in preds:
    if pred == 2:
        print("low")
    elif pred == 3:
        print("normal")
    elif pred == 1:
        print("high")
    else:
        print("heavy")
