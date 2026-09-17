"""Historical notebook cells [14, 31, 32, 37, 40, 48, 56, 71, 76, 82, 100, 101, 104, 111, 114, 115, 117, 131, 135, 153, 188, 213, 216, 220, 222, 226, 322, 323, 387, 403]; structural extraction, unchanged scientific logic."""
import pandas as pd
import numpy as np
import random
import re
import json
import ast
from pathlib import Path

def prepare_education(frames):
    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates(inplace=True)
    return df

def prepare_bbq(df):
    df = df[df['context_condition'] == 'ambig']
    sampled_df = df.groupby('category', group_keys=False).apply(lambda g: g.sample(n=91, random_state=42)).reset_index(drop=True)
    return sampled_df

def match_recruitment(df_cv, df_jobs):
    df_cv = df_cv.dropna(subset=['CV', 'Primary Keyword', 'Experience Years', 'English Level']).copy()
    df_cv['Experience Years'] = pd.to_numeric(df_cv['Experience Years'], errors='coerce').fillna(0)
    df_cv['Primary Keyword'] = df_cv['Primary Keyword'].str.lower().str.strip()
    df_cv['English Level'] = df_cv['English Level'].str.lower().str.strip()
    df_jobs = df_jobs.dropna(subset=['Long Description', 'Primary Keyword', 'Exp Years', 'English Level']).copy()
    df_jobs['Exp Years'] = df_jobs['Exp Years'].str.replace('y', '', regex=False)
    df_jobs['Exp Years'] = pd.to_numeric(df_jobs['Exp Years'], errors='coerce').fillna(0)
    df_jobs['Primary Keyword'] = df_jobs['Primary Keyword'].str.lower().str.strip()
    df_jobs['English Level'] = df_jobs['English Level'].str.lower().str.strip()
    sampled_pairs = []
    df_jobs_shuffled = df_jobs.sample(frac=1, random_state=42).reset_index(drop=True)
    import re
    for (_, job) in df_jobs_shuffled.iterrows():
        keyword = job['Primary Keyword']
        if keyword == 'other':
            continue
        level = job['English Level']
        exp_required = job['Exp Years']
        matched = df_cv[df_cv['Primary Keyword'].str.contains(re.escape(keyword), na=False) & (df_cv['Primary Keyword'] != 'other') & (df_cv['English Level'] == level) & df_cv['Experience Years'].between(exp_required, exp_required + 1)]
        if not matched.empty:
            candidate = matched.sample(1, random_state=random.randint(0, 10000)).iloc[0]
            sampled_pairs.append({'job_id': job.get('id', None), 'job_title': job['Position'], 'job_description': job['Long Description'], 'job_keyword': job['Primary Keyword'], 'job_exp_required': job['Exp Years'], 'job_english': job['English Level'], 'candidate_id': candidate.get('id', None), 'candidate_position': candidate['Position'], 'candidate_keyword': candidate['Primary Keyword'], 'candidate_exp': candidate['Experience Years'], 'candidate_english': candidate['English Level'], 'cv_text': candidate['CV']})
        if len(sampled_pairs) == 1000:
            break
    return pd.DataFrame(sampled_pairs)

def prepare_movielens(ratings, movies):
    ratings = ratings.merge(movies, on='movieId')
    unique_users = ratings['userId'].unique()
    sampled_users = ratings['userId'].drop_duplicates().sample(n=200, random_state=42)
    ratings_sampled = ratings[ratings['userId'].isin(sampled_users)]
    top_rated = ratings_sampled.sort_values(['userId', 'rating', 'timestamp'], ascending=[True, False, False]).groupby('userId').head(10).assign(anchor_type='top-rated')
    most_recent = ratings_sampled.sort_values(['userId', 'timestamp'], ascending=[True, False]).groupby('userId').head(10).assign(anchor_type='recent')
    anchors_combined = pd.concat([top_rated, most_recent], ignore_index=True)
    anchors_combined['year'] = anchors_combined['title'].str.extract('\\((\\d{4})\\)').astype(float)
    anchors_combined['genres_list'] = anchors_combined['genres'].str.split('|')
    liked = anchors_combined[anchors_combined['rating'] >= 4].copy()

    def summarize_user(group):
        min_year = int(group['year'].min())
        max_year = int(group['year'].max())
        genres = sorted(set((g for sublist in group['genres_list'] for g in sublist)))
        genre_str = ', '.join(genres)
        year_range = f'from the years {min_year} to {max_year}'
        genre_phrase = f'genres such as {genre_str}'
        return pd.Series({'year_range': year_range, 'genre_phrase': genre_phrase})
    user_profiles = liked.groupby('userId').apply(summarize_user).reset_index()
    anchors_combined = anchors_combined.merge(user_profiles, on='userId', how='left')

    def format_movie(row):
        title_clean = row['title'].rsplit(' (', 1)[0]
        return f"’{title_clean}’ ({int(row['year'])}, {row['genres']}, Rating: {row['rating']}/5)"
    anchors_combined['formatted_movie'] = anchors_combined.apply(format_movie, axis=1)
    user_anchor_groups = anchors_combined.groupby(['userId', 'anchor_type', 'year_range', 'genre_phrase'])['formatted_movie'].apply(lambda x: ', '.join(x)).reset_index(name='anchor_movies')
    return user_anchor_groups

def sample_biasmd(df):
    type_counts = df['Type'].value_counts()
    target_total = 1000
    proportional_counts = (type_counts / type_counts.sum() * target_total).round().astype(int)
    diff = proportional_counts.sum() - target_total
    if diff != 0:
        max_type = proportional_counts.idxmax()
        proportional_counts[max_type] -= diff
    df_sampled = df.groupby('Type', group_keys=False).apply(lambda x: x.sample(n=proportional_counts[x.name], random_state=42)).reset_index(drop=True)
    return df_sampled

def sample_medical_bias(merged_df):
    target_total = 1000
    bias_types = merged_df['bias_type'].unique()
    n_types = len(bias_types)
    samples_per_type = target_total // n_types + 1
    samples = []
    for bias in bias_types:
        df_subset = merged_df[merged_df['bias_type'] == bias]
        sampled = df_subset.sample(n=samples_per_type, random_state=42)
        samples.append(sampled)
    df_sampled = pd.concat(samples)
    df_sampled = df_sampled.sample(n=1000, random_state=42)
    df_sampled = df_sampled.reset_index(drop=True)
    print(df_sampled['bias_type'].value_counts())
    return df_sampled

def sample_disease_buster(df):
    bias_types = df['Type'].value_counts().index.tolist()
    samples_per_type = 31
    extra = 8
    samples = []
    for bias in bias_types:
        df_subset = df[df['Type'] == bias]
        sampled = df_subset.sample(n=samples_per_type, random_state=42)
        samples.append(sampled)
    extra_types = pd.Series(bias_types).sample(n=extra, random_state=42)
    for bias in extra_types:
        df_subset = df[df['Type'] == bias]
        sampled = df_subset.sample(n=1, random_state=42)
        samples.append(sampled)
    df_sampled = pd.concat(samples).reset_index(drop=True)
    print(df_sampled['Type'].value_counts())
    print('Total samples:', len(df_sampled))
    return df_sampled

def sample_mental_multilabel(df):
    from skmultilearn.model_selection import iterative_train_test_split
    import pandas as pd
    import numpy as np
    label_cols = ['Financial_Problem', 'Other', 'Everyday_Decision_Making', 'Emotional_Turmoil', 'School', 'Family_Issues', 'Social_Relationships', 'Work', 'Health_Fatigue_Physical_Pain']
    X = df[['text']].values
    y = df[label_cols].values
    (X_sampled, y_sampled, X_remaining, y_remaining) = iterative_train_test_split(X, y, test_size=(len(df) - 1000) / len(df))
    df_sampled = pd.DataFrame(np.hstack((X_sampled, y_sampled)), columns=['text'] + label_cols)
    df_remaining = pd.DataFrame(np.hstack((X_remaining, y_remaining)), columns=['text'] + label_cols)
    df_sampled[label_cols] = df_sampled[label_cols].astype(int)
    df_remaining[label_cols] = df_remaining[label_cols].astype(int)
    missing = 1000 - len(df_sampled)
    if missing > 0:
        pad_rows = df_remaining.sample(n=missing, random_state=42)
        df_sampled = pd.concat([df_sampled, pad_rows]).reset_index(drop=True)
        df_remaining = df_remaining.drop(pad_rows.index).reset_index(drop=True)
    return {'sampled': df_sampled, 'remaining': df_remaining}

def sample_mental_joint_labels(df):
    import pandas as pd
    df['joint_label'] = df['Thwarted_Belongingness'].astype(str) + df['Perceived_Burdensomeness'].astype(str)
    label_dist = df['joint_label'].value_counts(normalize=True)
    target_counts = (label_dist * 1000).round().astype(int)
    diff = 1000 - target_counts.sum()
    if diff != 0:
        target_counts.iloc[0] += diff
    samples = []
    for (label, count) in target_counts.items():
        subset = df[df['joint_label'] == label]
        sampled = subset.sample(n=count, random_state=42)
        samples.append(sampled)
    df_sampled = pd.concat(samples).reset_index(drop=True)
    df_sampled = df_sampled.drop(columns='joint_label')
    return df_sampled

def sample_mental_single_label(df):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    label_counts = df['label'].value_counts()
    total_count = label_counts.sum()
    label_fraction = label_counts / total_count
    test_counts = (label_fraction * 1000).round().astype(int)
    diff = 1000 - test_counts.sum()
    if diff != 0:
        largest_label = test_counts.idxmax()
        test_counts[largest_label] += diff
    test_df_parts = []
    train_df_parts = []
    for (label, count) in test_counts.items():
        label_df = df[df['label'] == label]
        test_part = label_df.sample(n=count, random_state=42)
        train_part = label_df.drop(test_part.index)
        test_df_parts.append(test_part)
        train_df_parts.append(train_part)
    test_df = pd.concat(test_df_parts)
    train_df = pd.concat(train_df_parts)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return {'test': test_df, 'train': train_df}

def sample_admission(df):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    sample_size = 1000
    (df_sampled, _) = train_test_split(df, train_size=sample_size, random_state=42)
    return df_sampled

def prepare_translation(pro_path, anti_path):
    df = pd.read_csv(pro_path, sep='\t', header=None, names=['gender', 'src_word_index', 'sentence', 'profession'])
    df2 = pd.read_csv(anti_path, sep='\t', header=None, names=['gender', 'src_word_index', 'sentence', 'profession'])
    df2['type'] = 'anti'
    df['type'] = 'pro'
    df = pd.concat([df, df2], ignore_index=True)
    return df

def attach_neutralized_text(df, responses_path):
    with open(responses_path) as f:
        data_list = [json.loads(line) for line in f if line.strip()]
    data = [item['response']['body']['choices'][0]['message']['content'] for item in data_list]
    df['neutralized_text'] = data
    return df

def update_legal_gender(csv_path, jsonl_path):
    df = pd.read_csv(csv_path)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            custom_id = entry.get('custom_id')
            response = entry.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
            if custom_id and response:
                try:
                    index = int(custom_id.split('-')[1])
                    df.loc[index, 'applicant_gender'] = int(response.strip())
                except (IndexError, ValueError):
                    continue
    return df

def prepare_ontonotes(input_path):
    jsonl_path = Path(input_path)
    records = []
    for line in jsonl_path.open():
        obj = json.loads(line)
        obj['instructions'] = json.dumps(obj['instructions'], separators=(',', ':'))
        records.append(obj)
    df = pd.DataFrame(records, columns=['text', 'instructions', 'sample_id', 'article_id', 'pair_id'])
    return df

def prepare_medbullets(df):

    def extract_age(text):
        age_match = re.search('\\b(\\d{1,3})\\b', text)
        age = int(age_match.group(1)) if age_match else None
        return age

    def extract_gender(text):
        text_lower = text.lower()
        if 'woman' in text_lower:
            gender = 'Female'
        elif 'girl' in text_lower:
            gender = 'Female'
        elif 'man' in text_lower:
            gender = 'Male'
        elif 'boy' in text_lower:
            gender = 'Male'
        elif 'male' in text_lower:
            gender = 'Male'
        elif 'female' in text_lower:
            gender = 'Male'
        else:
            gender = None
        return gender
    df['gender'] = df['question'].apply(extract_gender)
    df['age'] = df['question'].apply(extract_age)
    age_group = []
    for age in df['age']:
        if age is None:
            age_group.append(None)
        elif age < 18:
            age_group.append('Minor')
        elif 18 <= age <= 65:
            age_group.append('Adult')
        else:
            age_group.append('Senior')
    df['age_group'] = age_group
    return df

def sample_medbullets_by_gender(df):
    import pandas as pd
    target_counts = {'Male': 320, 'Female': 168, 'None': 12}
    samples = []
    for (gender, count) in target_counts.items():
        if gender == 'None':
            subset = df[pd.isna(df['gender'])]
        else:
            subset = df[df['gender'] == gender]
        sampled = subset.sample(n=count, random_state=42)
        samples.append(sampled)
    df_sampled = pd.concat(samples).reset_index(drop=True)
    print(df_sampled['gender'].value_counts(dropna=False))
    return df_sampled

def prepare_medical_prompts(base_dir):

    def collect_prompts(base_dir):
        """
    Walks through bias_type/no-mitigation/test folders and reads each text file.
    Returns a DataFrame with columns: 'bias_type' and 'prompt'.
    """
        base_path = Path(base_dir)
        rows = []
        for bias_dir in base_path.iterdir():
            if not bias_dir.is_dir():
                continue
            bias_type = bias_dir.name
            test_path = bias_dir / 'no-mitigation' / 'test'
            if not test_path.exists():
                continue
            for txt_file in test_path.glob('*.txt'):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                    rows.append({'bias_type': bias_type, 'prompt': prompt})
        df = pd.DataFrame(rows)
        return df
    df = collect_prompts(base_dir)
    df['prompt'] = df['prompt'].apply(lambda x: x.replace('GPT model', 'assistant'))
    return df

def extract_admission_fields(df):

    def extract_structured_fields(text):
        try:
            gender = re.search('The (\\w+) student', text).group(1)
            major = ast.literal_eval(re.search('studied (\\[.*?\\])', text).group(1))[0]
            school = re.search('at (.*?) with a GPA', text).group(1)
            gpa = float(re.search('GPA of ([\\d.]+)', text).group(1))
            degree = re.search('receiving the degree of (.+?)\\.', text).group(1)
            gre_str = re.search('GRE test with scores (.*?) and TOEFL', text).group(1)
            gre = ast.literal_eval(gre_str)
            gre_verbal = gre.get('Verbal')
            gre_quant = gre.get('Quantitative')
            gre_aw = gre.get('Analytical Writing')
            toefl_str = re.search('TOEFL test with scores (.*?)\\.', text).group(1)
            toefl = ast.literal_eval(toefl_str)
            toefl_total = toefl.get('Total')
            interest = ast.literal_eval(re.search('interested in (\\[.*?\\])', text).group(1))[0]
            country = re.search('are from ([^\\d]+?) and', text).group(1)
            age = int(re.search('are (\\d{1,3}) years old', text).group(1))
            rec_score = int(re.search('evaluation of (\\d+)/10', text).group(1))
            return pd.Series({'gender': gender, 'major': major, 'school': school, 'gpa': gpa, 'degree': degree, 'gre_verbal': gre_verbal, 'gre_quant': gre_quant, 'gre_aw': gre_aw, 'toefl_total': toefl_total, 'interest': interest, 'country': country, 'age': age, 'rec_score': rec_score})
        except Exception as e:
            print(text)
            print(f'Failed to parse prompt: {e}')
            return pd.Series([None] * 13, index=['gender', 'major', 'school', 'gpa', 'degree', 'gre_verbal', 'gre_quant', 'gre_aw', 'toefl_total', 'interest', 'country', 'age', 'rec_score'])
    extracted_df = df['prompts'].apply(extract_structured_fields)
    df = pd.concat([df, extracted_df], axis=1)
    return df
