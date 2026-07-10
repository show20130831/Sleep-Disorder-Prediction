import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Person ID": [1, 2, 3, 4],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Age": [27, 28, 45, 52],
            "Occupation": ["Nurse", "Doctor", "Nurse", "Teacher"],
            "Sleep Duration": [6.1, 7.0, 5.5, 8.0],
            "Quality of Sleep": [6, 8, 4, 9],
            "Physical Activity Level": [42, 60, 30, 70],
            "Stress Level": [6, 4, 8, 3],
            "BMI Category": ["Normal", "Normal Weight", "Obese", "Overweight"],
            "Blood Pressure": ["126/83", "125/80", "140/95", "118/76"],
            "Heart Rate": [77, 70, 85, 68],
            "Daily Steps": [4200, 8000, 3000, 9000],
            "Sleep Disorder": ["None", "None", "Sleep Apnea", "Insomnia"],
        }
    )
