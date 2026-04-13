"""Input validators — v1 schema (fertility classifier)."""
from typing import Any, Dict, List

VALID_MENSTRUAL = {"Regular", "Irregular"}
VALID_PCOS = {"Yes", "No"}
VALID_STRESS = {"Low", "Medium", "High"}
VALID_SMOKING = {"Yes", "No"}
VALID_ALCOHOL = {"None", "Moderate", "High"}
VALID_TREATMENT = {"None", "Medication", "IVF"}

REQUIRED_FIELDS = [
    "female_age", "male_age", "bmi", "menstrual_regularity", "pcos",
    "stress_level", "smoking", "alcohol_intake",
    "sperm_count_million_per_ml", "motility_pct",
    "trying_duration_months", "treatment_type",
]


def validate_record(record: Dict[str, Any]) -> List[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    if errors:
        return errors

    if not isinstance(record["female_age"], int) or not (18 <= record["female_age"] <= 50):
        errors.append("female_age must be an integer between 18 and 50")
    if not isinstance(record["male_age"], int) or not (18 <= record["male_age"] <= 70):
        errors.append("male_age must be an integer between 18 and 70")
    if not isinstance(record["bmi"], (int, float)) or not (10.0 <= record["bmi"] <= 60.0):
        errors.append("bmi must be a float between 10.0 and 60.0")
    if not isinstance(record["sperm_count_million_per_ml"], (int, float)) or record["sperm_count_million_per_ml"] < 0:
        errors.append("sperm_count_million_per_ml must be a non-negative number")
    if not isinstance(record["motility_pct"], (int, float)) or not (0 <= record["motility_pct"] <= 100):
        errors.append("motility_pct must be between 0 and 100")
    if not isinstance(record["trying_duration_months"], int) or record["trying_duration_months"] < 0:
        errors.append("trying_duration_months must be a non-negative integer")
    if record["menstrual_regularity"] not in VALID_MENSTRUAL:
        errors.append(f"menstrual_regularity must be one of {VALID_MENSTRUAL}")
    if record["pcos"] not in VALID_PCOS:
        errors.append(f"pcos must be one of {VALID_PCOS}")
    if record["stress_level"] not in VALID_STRESS:
        errors.append(f"stress_level must be one of {VALID_STRESS}")
    if record["smoking"] not in VALID_SMOKING:
        errors.append(f"smoking must be one of {VALID_SMOKING}")
    if record["alcohol_intake"] not in VALID_ALCOHOL:
        errors.append(f"alcohol_intake must be one of {VALID_ALCOHOL}")
    if record["treatment_type"] not in VALID_TREATMENT:
        errors.append(f"treatment_type must be one of {VALID_TREATMENT}")
    return errors
