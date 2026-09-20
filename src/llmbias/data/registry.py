from llmbias.processing.requests.mental_health import create_mental_health_template
from llmbias.processing.requests.medical import create_med_bullets_template, create_medical_bias_template
from llmbias.processing.requests.recommendation import create_rec_system_template
from llmbias.processing.requests.recruitment import create_djinni_template
from llmbias.processing.requests.education import create_education_template
from llmbias.processing.requests.translation import create_translation_template
from llmbias.processing.requests.legal import create_ecthr_template
from llmbias.processing.requests.summarization import create_ontonotes_template
from llmbias.processing.requests.conversational import create_bold_template, create_bbq_template

functions = {
    "CAMS": create_mental_health_template,
    "SAD": create_mental_health_template,
    "medbullets": create_med_bullets_template,
    "medical_bias": create_medical_bias_template,
    "movielens": create_rec_system_template,
    "djinni": create_djinni_template,
    "education_ranking": create_education_template,
    "mt_gender": create_translation_template,
    "ecthr": create_ecthr_template,
    "ontonotes": create_ontonotes_template,
    "bold": create_bold_template,
    "bbq": create_bbq_template,
}
