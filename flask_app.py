import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)

# Paths to your data files
real_data_path = 'realdata.csv'
encoded_data_path = 'encoded_data.csv'

# Load the real data
df_real = pd.read_csv(real_data_path)

# Function to preprocess multivalued features
def preprocess_multivalued(df, column_name, mlb=None):
    df[column_name] = df[column_name].fillna('')
    if mlb is None:
        mlb = MultiLabelBinarizer()
        expanded_data = mlb.fit_transform(df[column_name].str.split(', '))
    else:
        expanded_data = mlb.transform(df[column_name].str.split(', '))
    expanded_df = pd.DataFrame(expanded_data, columns=[f"{column_name}_{cls}" for cls in mlb.classes_], index=df.index)
    return expanded_df, mlb

# Initialize and preprocess the dataset
multivalued_features = ['skills', 'businessGoals', 'preferredPartnerIndustry']
mlb_dict = {}
df_encoded = df_real[['username', 'phone', 'email', 'companyName', 'location', 'industryType', 'experienceLevel', 'businessSize', 'businessStage']].copy()  # Include necessary columns

for feature in multivalued_features:
    expanded_df, mlb = preprocess_multivalued(df_real, feature)
    df_encoded = pd.concat([df_encoded, expanded_df], axis=1)
    mlb_dict[feature] = mlb

categorical_features = ['industryType', 'experienceLevel', 'businessSize', 'location', 'businessStage']
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
onehot_encoded = onehot_encoder.fit_transform(df_real[categorical_features])
df_encoded = pd.concat([df_encoded, pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_features))], axis=1)

numerical_features = ['connectionsMade', 'partnershipSuccessful']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_real[numerical_features])
df_encoded = pd.concat([df_encoded, pd.DataFrame(scaled_features, columns=numerical_features)], axis=1)

X = df_encoded.drop(columns=['username', 'phone', 'email', 'companyName', 'location', 'industryType', 'experienceLevel', 'businessSize', 'businessStage'])

# Handle missing values explicitly if they exist
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

kmeans = KMeans(n_clusters=6, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(X)

df_encoded.to_csv(encoded_data_path, index=False)

# Function to add a new user to the real data CSV
def add_new_user_to_realdata(new_user_data):
    df_real = pd.read_csv(real_data_path)
    new_user_df = pd.DataFrame([new_user_data])
    df_real = pd.concat([df_real, new_user_df], ignore_index=True)
    df_real.to_csv(real_data_path, index=False)
    return new_user_df

# Function to preprocess and encode new user data
def preprocess_new_user(new_user_data):
    new_user_df = pd.DataFrame([new_user_data])
    
    new_user_encoded = pd.DataFrame()
    
    for feature in multivalued_features:
        expanded_df, _ = preprocess_multivalued(new_user_df, feature, mlb_dict[feature])
        new_user_encoded = pd.concat([new_user_encoded, expanded_df], axis=1)
        
    onehot_encoded_new = onehot_encoder.transform(new_user_df[categorical_features])
    new_user_encoded = pd.concat([new_user_encoded, pd.DataFrame(onehot_encoded_new, columns=onehot_encoder.get_feature_names_out(categorical_features))], axis=1)
    
    scaled_features_new = scaler.transform(new_user_df[numerical_features])
    new_user_encoded = pd.concat([new_user_encoded, pd.DataFrame(scaled_features_new, columns=numerical_features)], axis=1)
    
    new_user_encoded = imputer.transform(new_user_encoded)  # Apply the same imputation for new user data
    
    return new_user_df, new_user_encoded

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
    
    new_user_encoded = imputer.transform(new_user_encoded)  # Apply the same imputation for new user data
    
    new_user_cluster = kmeans.predict(new_user_encoded)[0]
    
    same_cluster_users = df_encoded[df_encoded['Cluster'] == new_user_cluster]
    same_cluster_users = same_cluster_users[same_cluster_users['username'] != new_user_data['username']]
    
    # Ensure that only existing columns are included in the response
    columns_to_include = ['username', 'phone', 'email', 'companyName', 'location', 'industryType', 'experienceLevel', 'businessSize', 'businessStage', 'skills', 'businessGoals', 'preferredPartnerIndustry']
    
    columns_present = [col for col in columns_to_include if col in same_cluster_users.columns]
    
    if not columns_present:
        return [], []  # If no columns are present, return empty results
    
    top_5_similar = same_cluster_users.head(10)[columns_present]
    

    # Ensure no NaN values are present in the top 5 results
    top_5_similar = top_5_similar.dropna()
    
    # Enhanced logic for recommending potential partners
    industry_mapping = {
        'Technology': ['E-commerce', 'Finance', 'Healthcare', 'Logistics', 'Agriculture'],
        'Healthcare': ['Technology', 'Manufacturing', 'Logistics'],
        'Finance': ['Technology', 'E-commerce', 'Retail'],
        'Retail': ['E-commerce', 'Logistics', 'Manufacturing'],
        'E-commerce': ['Technology', 'Logistics', 'Retail'],
        'Manufacturing': ['Logistics', 'Technology', 'Healthcare', 'Retail'],
        'Management': ['Technology', 'Logistics', 'Finance'],
        'Logistics': ['E-commerce', 'Manufacturing', 'Retail'],
        'Tourism': ['Technology', 'Healthcare'],
        'Agriculture': ['Technology', 'Manufacturing', 'Logistics']
    }
    
    location_mapping = {
        'Chennai': ['Bangalore', 'Pondicherry', 'Hyderabad', 'Mumbai'],
        'Bangalore': ['Chennai', 'Hyderabad', 'Pune', 'Mysore'],
        'Hyderabad': ['Bangalore', 'Chennai', 'Pune', 'Mumbai'],
        'Pune': ['Mumbai', 'Bangalore', 'Hyderabad', 'Goa'],
        'Mumbai': ['Pune', 'Goa', 'Hyderabad', 'Delhi'],
        'Goa': ['Mumbai', 'Chennai', 'Pondicherry', 'Goa'],
        'Mysore': ['Bangalore', 'Chennai', 'Pondicherry', 'Goa'],
        'Pondicherry': ['Chennai', 'Bangalore', 'Mysore', 'Hyderabad'],
        'Delhi': ['Kolkata', 'Mumbai', 'Hyderabad', 'Pune'],
        'Kolkata': ['Delhi', 'Mumbai', 'Chennai', 'Hyderabad']
    }
    
    # Industry and location match
    potential_partners = df_real[
        (df_real['industryType'].apply(lambda x: x in industry_mapping.get(new_user_data['industryType'], []))) &
        (df_real['location'].apply(lambda x: x in location_mapping.get(new_user_data['location'], []))) &
        (df_real['experienceLevel'] != new_user_data['experienceLevel']) &
        (df_real['businessSize'] != new_user_data['businessSize'])
    ]
    
    potential_partners = potential_partners[potential_partners['username'] != new_user_data['username']]
    top_5_potential_partners = potential_partners.head(10)[columns_present]
    

    top_5_potential_partners = top_5_potential_partners.dropna()

    return top_5_similar.to_dict(orient='records'), top_5_potential_partners.to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    new_user_data = request.json
    
    # Assuming recommend_top_5_users is a function that takes in new_user_data
    top_5_similar, top_5_potential_partners = recommend_top_5_users(new_user_data)
    
    return jsonify({
        'top_5_similar': top_5_similar,
        'top_5_potential_partners': top_5_potential_partners
    })

if __name__ == '__main__':
    app.run(debug=True)
