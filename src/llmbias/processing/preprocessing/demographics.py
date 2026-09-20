import re

GENDER_PRONOUN_MAP = {
    'male': {
        'he': 'she',
        'him': 'her',
        'his': 'her',
        'himself': 'herself',
        'man': 'woman',
        'boy': 'girl',
        'male': 'female'
    },
    'female': {
        'she': 'he',
        'her': 'him',
        'hers': 'his',
        'herself': 'himself',
        'woman': 'man',
        'girl': 'boy',
        'female': 'male'
    }
}

GENDERS = ['male', 'female']

def remove_gender_mentions(text):
    """
    Replace gendered pronouns and terms with neutral phrases.
    """
    replacements = {
        r'\bhe\b': 'they',
        r'\bshe\b': 'they',
        r'\bhim\b': 'them',
        r'\bher\b': 'them',
        r'\bhis\b': 'their',
        r'\bhers\b': 'theirs',
        r'\bhimself\b': 'themselves',
        r'\bherself\b': 'themselves',
        r'\bman\b': 'person',
        r'\bwoman\b': 'person',
        r'\bboy\b': 'child',
        r'\bgirl\b': 'child',
        r'\bmale\b': 'individual',
        r'\bfemale\b': 'individual',
        r'\bgentleman\b': 'person',
        r'\blady\b': 'person',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def swap_gender(text, from_gender, to_gender):
    pronoun_map = GENDER_PRONOUN_MAP[from_gender]
    for src, tgt in pronoun_map.items():
        text = re.sub(rf'\b{src}\b', tgt, text, flags=re.IGNORECASE)
    return text

def add_ethnicity_to_gender(text, ethnicity):
    replacements = {
        'woman': f'{ethnicity} woman',
        'girl': f'{ethnicity} girl',
        'man': f'{ethnicity} man',
        'boy': f'{ethnicity} boy',
        'male': f'{ethnicity} male',
        'female': f'{ethnicity} female',
    }

    for word, replacement in replacements.items():
        # \b ensures whole word match; re.IGNORECASE handles casing
        text = re.sub(rf'\b{word}\b', replacement, text, flags=re.IGNORECASE)
    
    return text
