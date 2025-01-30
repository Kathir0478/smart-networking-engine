from flask import Flask, jsonify, request
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# Creating a Flask app
app = Flask(__name__)

# Paths to your data files
real_data_path = 'realdata.csv'
encoded_data_path = 'encoded_data.csv'

# Load the real data
df_real = pd.read_csv(real_data_path)

# Function to preprocess multivalued features
def preprocess_multivalued(df, column_name, mlb=None):
    if mlb is None:
        mlb = MultiLabelBinarizer()
        expanded_data = mlb.fit_transform(df[column_name].str.split(', '))
    else:
        expanded_data = mlb.transform(df[column_name].str.split(', '))
    expanded_df = pd.DataFrame(expanded_data, columns=[f"{column_name}_{cls}" for cls in mlb.classes_], index=df.index)
    return expanded_df, mlb

# Initialize and preprocess the dataset
multivalued_features = ['Skills', 'Business_Goals', 'Preferred_Partner_Industry']
mlb_dict = {}
df_encoded = df_real[['Entrepreneur_ID']].copy()

for feature in multivalued_features:
    expanded_df, mlb = preprocess_multivalued(df_real, feature)
    df_encoded = pd.concat([df_encoded, expanded_df], axis=1)
    mlb_dict[feature] = mlb

categorical_features = ['Industry', 'Experience_Level', 'Business_Size', 'Location', 'Business_Stage']
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded = onehot_encoder.fit_transform(df_real[categorical_features])
df_encoded = pd.concat([df_encoded, pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_features))], axis=1)

numerical_features = ['Connections_Made', 'Partnership_Successful']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_real[numerical_features])
df_encoded = pd.concat([df_encoded, pd.DataFrame(scaled_features, columns=numerical_features)], axis=1)

X = df_encoded.drop(columns=['Entrepreneur_ID'])
kmeans = KMeans(n_clusters=5, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(X)

df_encoded.to_csv(encoded_data_path, index=False)

df_merged = df_real[['Entrepreneur_ID', 'Name', 'Number', 'Mail']].merge(df_encoded, on='Entrepreneur_ID')

# Function to add a new user to the real data CSV
def add_new_user_to_realdata(new_user_data):
    df_real = pd.read_csv(real_data_path)
    new_user_df = pd.DataFrame([new_user_data])
    df_real = pd.concat([df_real, new_user_df], ignore_index=True)
    df_real.to_csv(real_data_path, index=False)
    return new_user_df

# Function to recommend top 5 users and potential partners
def recommend_top_5_users(new_user_data):
    new_user_df = add_new_user_to_realdata(new_user_data)
    new_user_encoded = pd.DataFrame()
    
    for feature in multivalued_features:
        expanded_df, _ = preprocess_multivalued(new_user_df, feature, mlb_dict[feature])
        new_user_encoded = pd.concat([new_user_encoded, expanded_df], axis=1)
        
    onehot_encoded_new = onehot_encoder.transform(new_user_df[categorical_features])
    new_user_encoded = pd.concat([new_user_encoded, pd.DataFrame(onehot_encoded_new, columns=onehot_encoder.get_feature_names_out(categorical_features))], axis=1)
    
    scaled_features_new = scaler.transform(new_user_df[numerical_features])
    new_user_encoded = pd.concat([new_user_encoded, pd.DataFrame(scaled_features_new, columns=numerical_features)], axis=1)
    
    new_user_cluster = kmeans.predict(new_user_encoded)[0]
    
    same_cluster_users = df_merged[df_merged['Cluster'] == new_user_cluster]
    same_cluster_users = same_cluster_users[same_cluster_users['Entrepreneur_ID'] != new_user_data['Entrepreneur_ID']]
    
    top_5_similar = same_cluster_users.head(5)[['Entrepreneur_ID', 'Name', 'Number', 'Mail']]
    
    industry_mapping = {
        'Technology': ['E-commerce', 'Finance', 'Healthcare', 'Logistics', 'Agriculture'],
        'Healthcare': ['Technology', 'Manufacturing', 'Logistics', 'Tourism'],
        'Finance': ['Technology', 'E-commerce', 'Retail', 'Healthcare'],
        'Retail': ['Ecommerce', 'Logistics', 'Manufacturing'],
        'Ecommerce': ['Technology', 'Logistics', 'Retail'],
        'Manufacturing': ['Logistics', 'Technology', 'Healthcare', 'Retail'],
        'Management': ['Technology', 'Logistics', 'Finance'],
        'Logistics': ['E-commerce', 'Manufacturing', 'Retail'],
        'Tourism': ['Technology', 'Healthcare'],
        'Agriculture': ['Technology', 'Manufacturing', 'Logistics']
    }
    
    potential_partners = df_real[
        (df_real['Industry'].apply(lambda x: x in industry_mapping[new_user_data['Industry']])) &
        (df_real['Experience_Level'] != new_user_data['Experience_Level']) &
        (df_real['Business_Size'] != new_user_data['Business_Size'])
    ]
    
    top_5_partners = potential_partners.head(5)[['Entrepreneur_ID', 'Name', 'Number', 'Mail']]
    
    return top_5_similar.to_dict(orient='records'), top_5_partners.to_dict(orient='records')

# Endpoint to get recommendations based on new user data
@app.route('/recommend', methods=['POST'])
def recommend():
    new_user_data = request.json
    top_5_similar_users, top_5_potential_partners = recommend_top_5_users(new_user_data)
    return jsonify({
        'Top_5_Similar_Users': top_5_similar_users,
        'Top_5_Potential_Partners': top_5_potential_partners
    })

# Driver function
if __name__ == '__main__':
    app.run(debug=True)
