from llmbias.tasks.mental_health import create_mental_health_template
from llmbias.tasks.medical import create_med_bullets_template, create_medical_bias_template
from llmbias.tasks.recommendation import create_rec_system_template
from llmbias.tasks.recruitment import create_djinni_template
from llmbias.tasks.education import create_education_template, create_education_admission_template
from llmbias.tasks.translation import create_translation_template
from llmbias.tasks.legal import create_ecthr_template
from llmbias.tasks.summarization import create_ontonotes_template
from llmbias.tasks.conversational import create_bold_template, create_bbq_template

functions = {
    "CAMS": create_mental_health_template,
    "dreaddit": create_mental_health_template,
    "SAD": create_mental_health_template,
    "medbullets": create_med_bullets_template,
    "medical_bias": create_medical_bias_template,
    "movielens": create_rec_system_template,
    "djinni": create_djinni_template,
    "education_ranking": create_education_template,
    "education_ga": create_education_admission_template,
    "mt_gender": create_translation_template,
    "ecthr": create_ecthr_template,
    "ontonotes": create_ontonotes_template,
    "bold": create_bold_template,
    "bbq": create_bbq_template,
}
