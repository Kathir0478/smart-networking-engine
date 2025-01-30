from faker import Faker
import random
import pandas as pd
import string
import bcrypt

fake = Faker()

industry_distribution = {
    'Technology': 0.25,
    'Healthcare': 0.20,
    'Finance': 0.15,
    'Retail': 0.10,
    'E-commerce': 0.10,
    'Manufacturing': 0.05,
    'Management': 0.05,
    'Logistics': 0.05,
    'Tourism': 0.03,
    'Agriculture': 0.02
}

experience_level_distribution = {
    'Novice': 0.40,
    'Intermediate': 0.35,
    'Expert': 0.25
}

business_size_distribution = {
    'Solo': 0.30,
    'Small Business': 0.40,
    'Medium Enterprise': 0.20,
    'Large Enterprise': 0.05,
    'Micro Business': 0.05
}

skills_list = ['Product Development', 'Marketing', 'Networking', 'Logistics', 'Management', 'Problem Solving', 'Emotional Intelligence', 'Innovation']

goals_list = ['Innovation', 'Market Expansion', 'Product Launch', 'Customer Acquisition', 'Service Development']

def generate_realistic_data(num_entries):
    data = []
    for _ in range(num_entries):
        entry = {
            '_id': fake.uuid4(),
            'username': fake.name(),
            'phone': random.randint(1000000000,9999999999),
            'email': fake.email(),
            'companyName': fake.company(),
            'password': bcrypt.hashpw((''.join(random.choices(string.ascii_letters + string.digits, k=10))).encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            'industryType': random.choices(list(industry_distribution.keys()), weights=industry_distribution.values())[0],
            'experienceLevel': random.choices(list(experience_level_distribution.keys()), weights=experience_level_distribution.values())[0],
            'businessSize': random.choices(list(business_size_distribution.keys()), weights=business_size_distribution.values())[0],
            'skills': ', '.join(random.sample(skills_list, random.randint(3, 5))),
            'location': random.choice(['Chennai', 'Banglore', 'Hyderabad', 'Pune', 'Mumbai', 'Goa', 'Mysore', 'Pondicherry', 'Delhi', 'Kolkata']),
            'businessGoals': ', '.join(random.sample(goals_list, random.randint(1, 2))),
            'businessStage': random.choice(['Startup', 'Growth', 'Mature']),
            'preferredPartnerIndustry': ', '.join(random.sample(list(industry_distribution.keys()), random.randint(1, 3))),
            'connectionsMade': random.randint(1, 100),
            'partnershipSuccessful': random.randint(1, 5),
        }
        data.append(entry)
    return data

num_entries = 233
entrepreneur_data = generate_realistic_data(num_entries)

df = pd.DataFrame(entrepreneur_data)
df.to_csv('realdata.csv', index=False)

print(f'Generated {num_entries} realistic entrepreneur records and saved to realistic_entrepreneur_data.csv')
