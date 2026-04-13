"""Unit tests -- input validators."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.validators import validate_record

VALID = {
    "female_age": 30, "male_age": 32, "bmi": 24.5,
    "menstrual_regularity": "Regular", "pcos": "No",
    "stress_level": "Medium", "smoking": "No",
    "alcohol_intake": "None", "sperm_count_million_per_ml": 55.0,
    "motility_pct": 65.0, "trying_duration_months": 12,
    "treatment_type": "None",
}

def test_valid_passes(): assert validate_record(VALID) == []
def test_missing_field(): assert any("female_age" in e for e in validate_record({k: v for k, v in VALID.items() if k != "female_age"}))
def test_invalid_pcos(): assert validate_record({**VALID, "pcos": "Maybe"}) != []
def test_invalid_treatment(): assert validate_record({**VALID, "treatment_type": "Surgery"}) != []
def test_age_out_of_range(): assert validate_record({**VALID, "female_age": 60}) != []
def test_motility_out_of_range(): assert validate_record({**VALID, "motility_pct": 110}) != []
def test_invalid_stress(): assert validate_record({**VALID, "stress_level": "Extreme"}) != []
def test_invalid_alcohol(): assert validate_record({**VALID, "alcohol_intake": "Low"}) != []
