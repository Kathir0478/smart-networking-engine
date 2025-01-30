import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import os

realdata_csv = 'realdata.csv'
encodeddata_csv = 'encodeddata.csv'

df = pd.read_csv(realdata_csv)

categorical_features = ['Industry', 'Experience_Level', 'Business_Size', 'Location', 'Business_Stage']
multivalued_features = ['Skills', 'Preferred_Partner_Industry', 'Business_Goals']
numerical_features = ['Connections_Made', 'Partnership_Successful']

def preprocess_multivalued_features(df, feature_names):
    mlb_dict = {}
    for feature in feature_names:
        mlb = MultiLabelBinarizer()
        df[feature] = df[feature].apply(lambda x: x.split(', ') if pd.notnull(x) else [])
        binarized = mlb.fit_transform(df[feature])
        binarized_columns = [f"{feature}_{label}" for label in mlb.classes_]
        df = pd.concat([df, pd.DataFrame(binarized, columns=binarized_columns)], axis=1)
        mlb_dict[feature] = mlb
    return df, mlb_dict

df, mlb_dict = preprocess_multivalued_features(df, multivalued_features)

remaining_categorical_features = [col for col in df.columns if col in categorical_features]
encoded_multivalued_features = [col for col in df.columns if col not in categorical_features + multivalued_features + numerical_features]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical = encoder.fit_transform(df[remaining_categorical_features])

scaler = StandardScaler()
X_numerical = scaler.fit_transform(df[numerical_features])

X_multivalued = df[encoded_multivalued_features].values
X = pd.concat([pd.DataFrame(X_categorical), pd.DataFrame(X_numerical), pd.DataFrame(X_multivalued)], axis=1)

kmeans = KMeans(n_clusters=20, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

df.to_csv(encodeddata_csv, index=False)
