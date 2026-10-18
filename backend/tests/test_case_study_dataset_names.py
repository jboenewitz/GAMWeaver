import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.ml_service import MLService


def test_case_study_upload_uses_exported_feature_names(monkeypatch, tmp_path):
    """The case-study option exposes the public names used by the supplied model export."""
    monkeypatch.setattr(
        MLService, "_restore_active_dataset_metadata", lambda self: None
    )
    monkeypatch.setattr(MLService, "_auto_load_persisted_dataset", lambda self: None)
    service = MLService()
    service.datasets_dir = tmp_path / "datasets"
    service.datasets_dir.mkdir()

    source_columns = [
        "Zielvariable – Lesekompetenz Deutsch",
        "Schulkontext während der Pandemie – Schulwochen im Distanzunterricht",
        "Sozioökonomischer Status – Höchster ISEI-Wert der Eltern",
        "Schüler:innenerleben – Soziale Integration (Skalenwert)",
        "Deutsch – Interesse (Skalenwert)",
        "Deutsch – Angst (Skalenwert)",
        "Deutsch – Wahrgenommener Leistungsdruck (Itemmittelwert 1–4)",
        "Digitale Kompetenzen – Computer- und Tabletkompetenz (Itemmittelwert)",
        "Deutschunterricht – Unterstützendes Lernklima (Itemmittelwert 1–4)",
        "Lernschwierigkeit – Lese- und Rechtschreibschwierigkeiten (1 = ja)",
        "Förderung – Sprach- oder Leseförderung erhalten (1 = ja)",
        "Familiäre Lesekultur – Bücher zu Hause (1–6)",
        "Sprachhintergrund – Zu Hause gesprochenes Deutsch (1 = nie, 3 = immer)",
    ]
    csv_bytes = pd.DataFrame([range(len(source_columns))], columns=source_columns).to_csv(
        index=False
    ).encode("utf-8")

    preview = service.inspect_uploaded_dataset(
        "case-study.csv", csv_bytes, use_case_study_feature_names=True
    )

    assert preview["columns"] == [
        "Zielvariable – Lesekompetenz Deutsch",
        "Schulkontext während der Pandemie – Schulwochen im Distanzunterricht",
        "Socioeconomic Status – Highest Parental ISEI Score",
        "Student Experience – Social Integration",
        "German – Interest",
        "German – Anxiety",
        "German – Perceived Performance Pressure",
        "Digital Competencies – Computer and Tablet Competence",
        "German Instruction – Supportive Learning Climate",
        "Learning Difficulty – Reading and Spelling Difficulties",
        "Förderung – Sprach- oder Leseförderung erhalten (1 = ja)",
        "Family Reading Culture – Books at Home",
        "Language Background – German Spoken at Home",
    ]
    assert preview["default_target_column"] == "Zielvariable – Lesekompetenz Deutsch"
    assert preview["case_study_feature_columns"] == [
        "Socioeconomic Status – Highest Parental ISEI Score",
        "Student Experience – Social Integration",
        "German – Interest",
        "German – Anxiety",
        "German – Perceived Performance Pressure",
        "Digital Competencies – Computer and Tablet Competence",
        "German Instruction – Supportive Learning Climate",
        "Learning Difficulty – Reading and Spelling Difficulties",
        "Family Reading Culture – Books at Home",
        "Language Background – German Spoken at Home",
    ]
    assert pd.read_csv(service.datasets_dir / preview["dataset_id"]).columns.tolist() == preview[
        "columns"
    ]
