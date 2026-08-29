#!/usr/bin/env python3
"""Erstellt Version 8 des deutschen IQB-Lesekompetenz-Datensatzes.

Die Zielvariable ist in beiden Exportvarianten der arithmetische Mittelwert aus
``PV_DLES_1`` bis ``PV_DLES_15``. Mit ``--feature-set`` wird gewählt, ob

* ``selected15``: 1 Zielvariable + die 14 speziell ausgewählten Prädiktoren
  (insgesamt 15 Spalten),
* ``all``: 1 Zielvariable + alle 99 in Version 4/6 ausgewählten Prädiktoren
  (insgesamt 100 Spalten), oder
* ``both``: beide kuratierten CSV-Dateien in einem Lauf, oder
* ``absolute-all``: Zielvariable plus alle nicht-imputierten Originalvariablen
  der SAV-Datei. Vollständige 15-fache Imputationsfamilien werden ausgelassen;
  beispielsweise wird ``Sind`` exportiert, nicht ``Sind_1`` bis ``Sind_15``.

ausgegeben werden.

Teilnehmende
------------
Es findet keine Filterung nach Geschlecht, Migrationshintergrund oder anderen
Personenmerkmalen statt. Stattdessen wird vor dem Export eine reproduzierbare,
einfache Zufallsstichprobe ohne Zurücklegen gezogen. Der gewünschte Anteil wird
beim Aufruf mit ``--sample-percent`` in Prozent angegeben, zum Beispiel
``--sample-percent 15``. Fehlende oder ungültige einzelne Merkmalswerte bleiben
als leere CSV-Zellen erhalten. Die Leerzellen bleiben bewusst bestehen, damit
die App fehlende Werte separat visualisieren kann statt sie mit 0 oder
Mittelwerten zu vermischen.

Für die kuratierten Feature-Sets werden grundsätzlich Originalvariablen
bevorzugt. Die drei Skalen ``Ssoe``, ``Sind`` und ``Sdeangst`` bleiben jedoch
wie in Version 7 auf den imputierten Variablen ``_1``.

Mit ``--random-seed`` lässt sich die Stichprobe reproduzieren oder gezielt
verändern. Bei ``--feature-set both`` werden für beide Ausgabedateien dieselben
Teilnehmenden ausgewählt. Für die beiden Schulkontextmerkmale zum Distanz- und
Wechselunterricht werden die nicht-imputierten Originalwerte aus der
Schulleitungs-SAV über ``IDSCH_FDZ`` ergänzt. Die Datei wird automatisch im
gleichen Ordner gesucht oder mit ``--principal-sav`` angegeben.

Abhängigkeiten
--------------
    python -m pip install pandas numpy pyreadstat

Beispiele
---------
Nur die 15 ausgewählten Spalten:

    python sav_to_csv_v8_de_random_percent.py input.sav \
      --feature-set selected15 --sample-percent 15

Alle 100 Spalten:

    python sav_to_csv_v8_de_random_percent.py input.sav \
      --feature-set all --sample-percent 15

Beide kuratierten Dateien:

    python sav_to_csv_v8_de_random_percent.py input.sav \
      --feature-set both --sample-percent 15

Alle nicht-imputierten Originalvariablen:

    python sav_to_csv_v8_de_random_percent.py input.sav \
      --feature-set absolute-all --sample-percent 15
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyreadstat  # type: ignore
except ImportError:  # --list-features remains usable without pyreadstat.
    pyreadstat = None


SCRIPT_VERSION = "8.1-de-random-percent-imputed-scales-pv-mean"
TOTAL_ALL_PREDICTORS = 99
TOTAL_ALL_OUTPUT_COLUMNS = 100
TOTAL_SELECTED_PREDICTORS = 14
TOTAL_SELECTED_OUTPUT_COLUMNS = 15
DEFAULT_RANDOM_SEED = 42
SCHOOL_JOIN_KEY = "IDSCH_FDZ"
PRINCIPAL_ORIGINAL_COLUMNS = frozenset({"Pdisbt_a_sum", "Pdisbt_b_sum"})

# Reihenfolge entspricht der vom Nutzer festgelegten 14er-Auswahl.
SELECTED15_GENERATION_STATUS_OUTPUT_NAME = (
    "Zuwanderung – Generationenstatus (EFB prioritär + SFB) [KMhgen]"
)
SELECTED15_ALIASES: dict[str, str] = {
    SELECTED15_GENERATION_STATUS_OUTPUT_NAME: (
        "Migrationshintergrund – Nähe zur eigenen Migrationserfahrung (0–4)"
    ),
}

SELECTED15_OUTPUT_NAMES: tuple[str, ...] = (
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
    "Sozioökonomischer Status – Soziale Klasse der Familie (höher = stärker begünstigt)",
    SELECTED15_GENERATION_STATUS_OUTPUT_NAME,
)


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of one output column."""

    output_name: str
    role: str
    category: str
    sources: tuple[str, ...]
    transform: str
    rationale: str
    min_valid: int | None = None
    prefer_non_imputed: bool = True


def F(
    output_name: str,
    category: str,
    sources: str | Sequence[str],
    transform: str = "raw",
    rationale: str = "",
    min_valid: int | None = None,
    prefer_non_imputed: bool = True,
) -> FeatureSpec:
    """Compact constructor for predictor specifications."""
    if isinstance(sources, str):
        source_tuple = (sources,)
    else:
        source_tuple = tuple(sources)
    return FeatureSpec(
        output_name=output_name,
        role="predictor",
        category=category,
        sources=source_tuple,
        transform=transform,
        rationale=rationale,
        min_valid=min_valid,
        prefer_non_imputed=prefer_non_imputed,
    )


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------


def pv_sources(base: str) -> tuple[str, ...]:
    return tuple(f"{base}_{number}" for number in range(1, 16))


TARGET_PV_MEAN = FeatureSpec(
    output_name="Zielvariable – Lesekompetenz Deutsch",
    role="target",
    category="Target",
    sources=pv_sources("PV_DLES"),
    transform="pv_mean",
    rationale=(
        "Arithmetischer Mittelwert aus PV_DLES_1 bis PV_DLES_15; "
        "alle 15 Werte müssen gültig vorliegen."
    ),
    min_valid=15,
)


# ---------------------------------------------------------------------------
# 99 predictors selected for German reading competence
# ---------------------------------------------------------------------------

PREDICTOR_SPECS: tuple[FeatureSpec, ...] = (
    # 1-12: Prior achievement, cognitive prerequisites and support
    F(
        "Vorleistung – Deutschnote (höherer Wert = bessere Leistung)",
        "Prior achievement and cognitive prerequisites",
        "TR_NOTE_D",
        "grade_higher_better",
        "Prior German achievement is a strong direct precursor of later reading performance.",
    ),
    F(
        "Vorleistung – Mathematiknote (höherer Wert = bessere Leistung)",
        "Prior achievement and cognitive prerequisites",
        "TR_NOTE_M",
        "grade_higher_better",
        "Mathematics grade provides a broader indicator of general school achievement.",
    ),
    F(
        "Kognitive Voraussetzung – Figurales Schlussfolgern (WLE-Wert)",
        "Prior achievement and cognitive prerequisites",
        "Tbefki_wle",
        rationale="General reasoning supports comprehension and complex task processing.",
    ),
    F(
        "Kognitive Voraussetzung – Wortschatzwert",
        "Prior achievement and cognitive prerequisites",
        "Tkftv_sum",
        rationale="Vocabulary is directly relevant to understanding written language.",
    ),
    F(
        "Kognitive Voraussetzung – Lesegeschwindigkeitswert",
        "Prior achievement and cognitive prerequisites",
        "Tlga_diffscore",
        rationale="Reading fluency is a central prerequisite for text comprehension.",
    ),
    F(
        "Schulübergang – Gymnasialempfehlung (1 = ja)",
        "Prior achievement and cognitive prerequisites",
        "TR_EMPFEHLUNG",
        "gym_recommendation",
        "Teacher recommendation summarises prior academic development and expectations.",
    ),
    F(
        "Lernschwierigkeit – Lese- und Rechtschreibschwierigkeiten (1 = ja)",
        "Prior achievement and cognitive prerequisites",
        "TR_TLS1",
        rationale="Documented reading-spelling difficulty is directly relevant to reading competence.",
    ),
    F(
        "Lernschwierigkeit – Isolierte Leseschwierigkeit (1 = ja)",
        "Prior achievement and cognitive prerequisites",
        "TR_TLS3",
        rationale="Isolated reading difficulty is a direct risk marker for reading competence.",
    ),
    F(
        "Lernschwierigkeit – Isolierte Rechtschreibschwierigkeit (1 = ja)",
        "Prior achievement and cognitive prerequisites",
        "TR_TLS2",
        rationale="Spelling difficulty captures related written-language processing problems.",
    ),
    F(
        "Förderung – Sprach- oder Leseförderung erhalten (1 = ja)",
        "Prior achievement and cognitive prerequisites",
        "TR_BES_FOE1",
        rationale="Indicates identified support needs and exposure to targeted language support.",
    ),
    F(
        "Förderbedarf – Sonderpädagogischer Förderbedarf (1 = ja)",
        "Prior achievement and cognitive prerequisites",
        "TR_SPF",
        rationale="Documented special educational needs can strongly affect learning trajectories.",
    ),
    F(
        "Lehrplan – Unterricht nach allgemeinem Deutschlehrplan (1 = ja)",
        "Prior achievement and cognitive prerequisites",
        "TR_ZIELGLEICH_D",
        rationale="Curriculum alignment determines the learning objectives used in German lessons.",
    ),

    # 13-25: Demographics, early schooling and school context
    F(
        "Demografie – Alter in Jahren",
        "Demographics and school context",
        "KAlter",
        rationale="Captures developmental and school-entry timing differences.",
    ),
    F(
        "Demografie – Geschlecht (0 = männlich, 1 = weiblich)",
        "Demographics and school context",
        "KGender",
        rationale="Standard demographic characteristic relevant to reading-performance differences.",
    ),
    F(
        "Frühkindliche Bildung – Dauer der vorschulischen Betreuung (0–6)",
        "Demographics and school context",
        "Ebekg16",
        rationale="Preschool participation can affect early language and literacy opportunities.",
    ),
    F(
        "Schuleintritt – Vorzeitige Einschulung (1 = ja)",
        "Demographics and school context",
        "Eschvz01a",
        rationale="Early entry reflects developmental timing and educational decisions.",
    ),
    F(
        "Schuleintritt – Zurückgestellte Einschulung (1 = ja)",
        "Demographics and school context",
        "Eschvz02a",
        rationale="Delayed entry can reflect developmental or support-related differences.",
    ),
    F(
        "Schulkontext – Förderschule (1 = Förderschule)",
        "Demographics and school context",
        "TR_SCHULFORM",
        "special_school_binary",
        "School form captures major differences in curricula and student composition.",
    ),
    F(
        "Schulkontext – Privatschule (1 = ja)",
        "Demographics and school context",
        "TR_PRIVAT",
        rationale="Private-school status may reflect differences in composition and resources.",
    ),
    F(
        "Schulkontext – Urbanisierungsgrad (1 = gering, 3 = hoch)",
        "Demographics and school context",
        "TR_URBAN",
        rationale="Urbanisation may relate to resources, composition and educational opportunities.",
    ),
    F(
        "Schulbiografie – Schulbesuch im Ausland in Jahren",
        "Demographics and school context",
        "Esaja",
        rationale="Schooling abroad can affect continuity and alignment with German instruction.",
    ),
    F(
        "Schulbiografie – Zusätzliche Monate Schulbesuch im Ausland",
        "Demographics and school context",
        "Esamo",
        rationale="Complements years abroad with a more precise duration measure.",
    ),
    F(
        "Migration – Alter bei der Ankunft in Deutschland",
        "Demographics and school context",
        "Sagezud_r",
        rationale="Later arrival generally implies less exposure to German-language schooling.",
    ),
    F(
        "Schulkontext während der Pandemie – Schulwochen im Distanzunterricht",
        "Demographics and school context",
        "Pdisbt_a_sum",
        rationale="School closure duration is relevant to the 2021 learning context.",
    ),
    F(
        "Schulkontext während der Pandemie – Schulwochen im Wechselunterricht",
        "Demographics and school context",
        "Pdisbt_b_sum",
        rationale="Alternating attendance altered instructional time and continuity.",
    ),

    # 26-43: Socioeconomic status and home educational capital
    F(
        "Familiäre Lesekultur – Bücher zu Hause (1–6)",
        "Socioeconomic and home literacy resources",
        "KBuecher",
        rationale="Books at home are a compact indicator of cultural and educational resources.",
    ),
    F(
        "Sozioökonomischer Status – Höchster ISEI-Wert der Eltern",
        "Socioeconomic and home literacy resources",
        "KHISEI",
        rationale="Highest parental occupational status captures socioeconomic resources.",
    ),
    F(
        "Elterliche Bildung – Höchste Bildungsdauer in Jahren",
        "Socioeconomic and home literacy resources",
        "EPARED",
        rationale="Parental education captures educational resources and expectations.",
    ),
    F(
        "Elterliche Bildung – Bildungsdauer der Mutter in Jahren",
        "Socioeconomic and home literacy resources",
        "EPAREDm",
        rationale="Mother's education provides a detailed home-education indicator.",
    ),
    F(
        "Elterliche Bildung – Bildungsdauer des Vaters in Jahren",
        "Socioeconomic and home literacy resources",
        "EPAREDf",
        rationale="Father's education provides a detailed home-education indicator.",
    ),
    F(
        "Elterliche Bildung – Höchster ISCED-Wert der Familie (0–6)",
        "Socioeconomic and home literacy resources",
        "EHISCED",
        rationale="Highest ISCED provides an ordinal indicator of formal educational attainment.",
    ),
    F(
        "Elterliche Bildung – ISCED-Wert der Mutter",
        "Socioeconomic and home literacy resources",
        "EISCEDm",
        rationale="Mother's formal qualification level may relate to literacy support at home.",
    ),
    F(
        "Elterliche Bildung – ISCED-Wert des Vaters",
        "Socioeconomic and home literacy resources",
        "EISCEDf",
        rationale="Father's formal qualification level may relate to literacy support at home.",
    ),
    F(
        "Sozioökonomischer Status – Beruflicher ISEI-Wert der Mutter",
        "Socioeconomic and home literacy resources",
        "Eberufm_ISEI",
        rationale="Mother's occupational status adds information beyond the family maximum.",
    ),
    F(
        "Sozioökonomischer Status – Beruflicher ISEI-Wert des Vaters",
        "Socioeconomic and home literacy resources",
        "Eberuff_ISEI",
        rationale="Father's occupational status adds information beyond the family maximum.",
    ),
    F(
        "Sozioökonomischer Status – Soziale Klasse der Familie (höher = stärker begünstigt)",
        "Socioeconomic and home literacy resources",
        "EHEGP6",
        "reverse_1_to_6",
        "Reverses the six EGP classes so larger values indicate greater social advantage.",
    ),
    F(
        "Familiäre Lesekultur – Von der Mutter gelesene literarische Werke (1–6)",
        "Socioeconomic and home literacy resources",
        "Elitm",
        rationale="Parental literary reading indicates literacy-related cultural capital.",
    ),
    F(
        "Familiäre Lesekultur – Vom Vater gelesene literarische Werke (1–6)",
        "Socioeconomic and home literacy resources",
        "Elitf",
        rationale="Parental literary reading indicates literacy-related cultural capital.",
    ),
    F(
        "Familiäre Lesekultur – Höchster elterlicher Wert zum Lesen literarischer Werke (1–6)",
        "Socioeconomic and home literacy resources",
        "KElitfam",
        rationale="Family maximum summarises the strongest literary-reading resource in the home.",
    ),
    F(
        "Elterliche Schulwahl – Bedeutung eines guten Rufs der Schule (1–4)",
        "Socioeconomic and home literacy resources",
        "Egswich01b",
        rationale="School-choice priorities can proxy educational aspirations and information use.",
    ),
    F(
        "Elterliche Schulwahl – Bedeutung hoher Schülerleistungen (1–4)",
        "Socioeconomic and home literacy resources",
        "Egswich01i",
        rationale="Importance placed on achievement may reflect academic expectations.",
    ),
    F(
        "Elterliche Schulwahl – Bedeutung einer guten Schulausstattung (1–4)",
        "Socioeconomic and home literacy resources",
        "Egswich01l",
        rationale="Attention to school resources may reflect educational investment.",
    ),
    F(
        "Elterliche Schulwahl – Bedeutung einer hohen Unterrichtsqualität (1–4)",
        "Socioeconomic and home literacy resources",
        "Egswich01m",
        rationale="Attention to teaching quality may reflect educational expectations.",
    ),

    # 44-57: German-language and migration background
    F(
        "Sprachhintergrund – Zu Hause gesprochenes Deutsch (1 = nie, 3 = immer)",
        "Language and migration background",
        "KDezh16_r",
        "home_german_higher",
        "Frequency of German use at home is directly relevant to German-language exposure.",
    ),
    F(
        "Sprachhintergrund – Altersgruppe beim Erwerb der deutschen Sprache (1–4)",
        "Language and migration background",
        "Esplern01a_r",
        "german_learning_age",
        "Age of German acquisition can show nonlinear associations with reading development.",
    ),
    F(
        "Sprachhintergrund – Deutsch als Muttersprache (1 = ja)",
        "Language and migration background",
        "TR_LANGUAGE",
        rationale="School-reported mother tongue provides a direct language-background indicator.",
    ),
    F(
        "Sprachhintergrund – Deutsch vor Schuleintritt gesprochen (1 = ja)",
        "Language and migration background",
        "Sspse16a_r",
        rationale="German exposure before school entry is relevant to early literacy development.",
    ),
    F(
        "Sprachhintergrund – Muttersprache der Mutter ist Deutsch (1 = ja)",
        "Language and migration background",
        "Emspm21a_r",
        rationale="Parental language background affects the home language environment.",
    ),
    F(
        "Sprachhintergrund – Muttersprache des Vaters ist Deutsch (1 = ja)",
        "Language and migration background",
        "Emspf21a_r",
        rationale="Parental language background affects the home language environment.",
    ),
    F(
        "Migrationshintergrund – Nähe zur eigenen Migrationserfahrung (0–4)",
        "Language and migration background",
        "KMhgen",
        "migration_proximity",
        "Reorders generation categories into an interpretable proximity scale.",
    ),
    F(
        "Migrationshintergrund – Schüler:in selbst zugewandert (1 = ja)",
        "Language and migration background",
        "KMigc",
        rationale="Own immigration experience can affect language exposure and schooling continuity.",
    ),
    F(
        "Migrationshintergrund – Mutter zugewandert (1 = ja)",
        "Language and migration background",
        "KMigm",
        rationale="Mother's immigration background contributes to the family language context.",
    ),
    F(
        "Migrationshintergrund – Vater zugewandert (1 = ja)",
        "Language and migration background",
        "KMigf",
        rationale="Father's immigration background contributes to the family language context.",
    ),
    F(
        "Migrationshintergrund – Anzahl der im Ausland geborenen Elternteile (0–2)",
        "Language and migration background",
        "KMigelt",
        rationale="Provides a gradual measure of parental migration background.",
    ),
    F(
        "Schüler:in als Geflüchtete:r nach Deutschland gekommen (Schulangabe)",
        "Language and migration background",
        "TR_FLUCHT",
        rationale=(
            "School-reported refugee background can indicate interrupted schooling "
            "and additional language-support needs."
        ),
    ),
    F(
        "Sprachhintergrund – Deutsch zu Hause laut Elternangabe (1 = nie, 3 = immer)",
        "Language and migration background",
        "Edezh16_r",
        "home_german_higher",
        "Retains the parent-specific report as an additional language-exposure measure.",
    ),
    F(
        "Sprachhintergrund – Deutsch zu Hause laut Schüler:innenangabe (1 = nie, 3 = immer)",
        "Language and migration background",
        "Sdezh16_r",
        "home_german_higher",
        "Retains the student-specific report as an additional language-exposure measure.",
    ),

    # 58-75: Motivation, affect, self-regulation and social context
    F(
        "Schüler:innenerleben – Schulzufriedenheit (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Salgsf",
        rationale="School satisfaction can support engagement and participation.",
    ),
    F(
        "Schüler:innenerleben – Soziale Integration (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Ssoe_1",
        rationale="Social integration may support well-being and classroom participation.",
        prefer_non_imputed=False,
    ),
    F(
        "Deutsch – Fachbezogenes Selbstkonzept (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Sskde",
        rationale="Subject-specific self-concept is closely connected to engagement and achievement.",
    ),
    F(
        "Deutsch – Interesse (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Sind_1",
        rationale="Interest supports sustained engagement with German-language learning.",
        prefer_non_imputed=False,
    ),
    F(
        "Deutsch – Angst (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Sdeangst_1",
        rationale="Anxiety can inhibit learning and performance in German tasks.",
        prefer_non_imputed=False,
    ),
    F(
        "Deutsch – Sorgenbezogene Angst (Itemmittelwert 1–4)",
        "Motivation, affect and self-regulation",
        ("Sword01a", "Sword01b", "Sword01c", "Sword01d"),
        "mean",
        "Separates cognitive worry from emotional arousal.",
        2,
    ),
    F(
        "Deutsch – Emotionale Angst (Itemmittelwert 1–4)",
        "Motivation, affect and self-regulation",
        ("Semod01a", "Semod01b", "Semod01c", "Semod01d"),
        "mean",
        "Captures emotional arousal in German learning and test situations.",
        2,
    ),
    F(
        "Persönliche Ressourcen – Allgemeine Selbstwirksamkeit (Itemmittelwert 1–4)",
        "Motivation, affect and self-regulation",
        ("Sswall01a", "Sswall01b", "Sswall01c", "Sswall01d"),
        "mean",
        "General self-efficacy supports persistence when tasks are difficult.",
        2,
    ),
    F(
        "Persönliche Ressourcen – Verantwortungsbewusstsein und Zuverlässigkeit (Itemmittelwert 1–4)",
        "Motivation, affect and self-regulation",
        ("SPersS21a", "SPersS21b", "SPersS21c", "SPersS21d"),
        "mean",
        "Responsibility and reliability proxy self-regulated learning behaviour.",
        2,
    ),
    F(
        "Deutsch – Wahrgenommener Leistungsdruck (Itemmittelwert 1–4)",
        "Motivation, affect and self-regulation",
        ("Sdruckd01a", "Sdruckd01b", "Sdruckd01c", "Sdruckd01d", "Sdruckd01e"),
        "mean",
        "Perceived pressure and workload may affect engagement and anxiety.",
        3,
    ),
    F(
        "Elternangabe – Schulischer Leistungsdruck (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Edrucks",
        rationale="Parent-reported pressure provides an additional perspective on workload.",
    ),
    F(
        "Lernen mit Gleichaltrigen – Erhaltene schulische Unterstützung (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Saudm",
        rationale="Support from classmates can help students overcome learning difficulties.",
    ),
    F(
        "Lernen mit Gleichaltrigen – Geleistete schulische Unterstützung (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Sauvm",
        rationale="Helping peers can reflect engagement and mastery-oriented participation.",
    ),
    F(
        "Schüler:innenpartizipation – Beteiligung an schulischen Entscheidungen (Itemmittelwert)",
        "Motivation, affect and self-regulation",
        ("SmitbezS21a", "SmitbezS21b", "SmitbezS21c", "SmitbezS21d"),
        "mean",
        "Participation may relate to school belonging and engagement.",
        2,
    ),
    F(
        "Digitale Kompetenzen – Computer- und Tabletkompetenz (Itemmittelwert)",
        "Motivation, affect and self-regulation",
        ("Sumz21a", "Sumz21b", "Sumz21c", "Sumz21d", "Sumz21e"),
        "mean",
        "Digital operational skills were important for accessing learning in 2021.",
        3,
    ),
    F(
        "Digitale Nutzung – Interesse und Selbstvertrauen im Umgang mit Geräten (Itemmittelwert)",
        "Motivation, affect and self-regulation",
        ("Semz21a", "Semz21b", "Semz21c", "Semz21d"),
        "mean",
        "Digital engagement can influence participation in technology-supported learning.",
        2,
    ),
    F(
        "Deutsch – Geschlechterrollenstereotype laut Schüler:innenangabe (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Sdegero",
        rationale="Subject stereotypes can interact with gender, interest and self-concept.",
    ),
    F(
        "Deutsch – Geschlechterrollenstereotype laut Elternangabe (Skalenwert)",
        "Motivation, affect and self-regulation",
        "Edegero",
        rationale="Parental stereotypes may shape expectations and learning support.",
    ),

    # 76-88: Reading and German instruction
    F(
        "Leseunterricht – Kognitive Aktivierung (Skalenwert)",
        "Reading and German instruction",
        "Skal",
        rationale="Captures cognitively demanding reading instruction.",
    ),
    F(
        "Leseunterricht – Bedeutung eines Textes erklären (1–4)",
        "Reading and German instruction",
        "Skal01a",
        rationale="Directly reflects meaning-focused text work.",
    ),
    F(
        "Leseunterricht – Vertiefende Fragen zu einem Text beantworten (1–4)",
        "Reading and German instruction",
        "Skal01b",
        rationale="Reflects deeper processing rather than simple retrieval.",
    ),
    F(
        "Leseunterricht – Ein neues Ende für eine Geschichte entwickeln (1–4)",
        "Reading and German instruction",
        "Skal01c",
        rationale="Reflects generative and elaborative text processing.",
    ),
    F(
        "Leseunterricht – Eigene Meinung zu einem Text äußern (1–4)",
        "Reading and German instruction",
        "Skal01d",
        rationale="Reflects evaluation and active engagement with text content.",
    ),
    F(
        "Leseunterricht – Eine Geschichte mit dem eigenen Leben verknüpfen (1–4)",
        "Reading and German instruction",
        "Skal01e",
        rationale="Activates personal knowledge and elaborative comprehension.",
    ),
    F(
        "Leseunterricht – Eine Geschichte mit Vorwissen verknüpfen (1–4)",
        "Reading and German instruction",
        "Skal01f",
        rationale="Activation of prior knowledge supports comprehension.",
    ),
    F(
        "Deutschunterricht – Unterstützendes Lernklima (Itemmittelwert 1–4)",
        "Reading and German instruction",
        ("Sulkd01a", "Sulkd01b", "Sulkd01c", "Sulkd01d", "Sulkd01e", "Sulkd01f", "Sulkd01g"),
        "mean",
        "Teacher support, encouragement and formative guidance can aid learning.",
        4,
    ),
    F(
        "Deutschunterricht – Klassenführung (Itemmittelwert 1–4)",
        "Reading and German instruction",
        ("Sklafd01a", "Sklafd01b", "Sklafd01c", "Sklafd01d", "Sklafd01e", "Sklafd01f"),
        "mean",
        "Orderly lessons increase effective learning time.",
        3,
    ),
    F(
        "Deutsch-Distanzunterricht – Kognitive Aktivierung",
        "Reading and German instruction",
        "Ldisau01d_D",
        rationale="Teacher-reported cognitive activation during distance instruction.",
    ),
    F(
        "Deutsch-Distanzunterricht – Motivation der Schüler:innen",
        "Reading and German instruction",
        "Ldisau01e_D",
        rationale="Teacher-reported focus on maintaining student motivation.",
    ),
    F(
        "Deutsch-Distanzunterricht – Berücksichtigung individueller Lernvoraussetzungen",
        "Reading and German instruction",
        "Ldisau01f_D",
        rationale="Teacher-reported adaptation to individual learning needs.",
    ),
    F(
        "Deutsch-Distanzunterricht – Erfassung des Lernfortschritts",
        "Reading and German instruction",
        "Ldisau01c_D",
        rationale="Teacher-reported monitoring of student learning progress.",
    ),

    # 89-99: Pandemic and home digital-learning conditions
    F(
        "Distanzlernen – Funktionieren aus Elternsicht (Skalenwert)",
        "Pandemic and digital-learning conditions",
        "Edisfunk",
        rationale="Parent assessment of how well distance learning functioned.",
    ),
    F(
        "Distanzlernen – Funktionieren aus Sicht der Deutschlehrkraft (Skalenwert)",
        "Pandemic and digital-learning conditions",
        "Ldisfunk_D",
        rationale="Teacher assessment of communication, motivation and implementation.",
    ),
    F(
        "Distanzlernen – Anteil des Deutschunterrichts in Präsenz (1–10)",
        "Pandemic and digital-learning conditions",
        "Ldisap_D",
        rationale="Ordinal share of German instruction delivered in person.",
    ),
    F(
        "Distanzlernen – Elterliche Unterstützung im Frühjahr 2020",
        "Pandemic and digital-learning conditions",
        "Edisunter01a",
        rationale="Amount of parental support during the first school-closure phase.",
    ),
    F(
        "Distanzlernen – Elterliche Unterstützung im Schuljahr 2020/21",
        "Pandemic and digital-learning conditions",
        "Edisunter01b",
        rationale="Amount of parental support during the school year studied.",
    ),
    F(
        "Häusliche Lernumgebung – Ruhiger Lernraum verfügbar (1 = ja)",
        "Pandemic and digital-learning conditions",
        "Edislumg02a",
        rationale="A quiet room supports sustained concentration during home learning.",
    ),
    F(
        "Häusliche Lernumgebung – Eigener Schreibtisch verfügbar (1 = ja)",
        "Pandemic and digital-learning conditions",
        "Edislumg02b",
        rationale="A dedicated workspace supports organised home learning.",
    ),
    F(
        "Häusliche Lernumgebung – Eigenes geeignetes Endgerät verfügbar (1 = ja)",
        "Pandemic and digital-learning conditions",
        "Edislumg02c",
        rationale="Device access was necessary for digital learning participation.",
    ),
    F(
        "Häusliche Lernumgebung – Ausreichende Internetverbindung (1 = ja)",
        "Pandemic and digital-learning conditions",
        "Edislumg02d",
        rationale="Reliable internet was necessary for digital learning participation.",
    ),
    F(
        "Distanzlernen – Deutschlehrkraft hielt Kontakt",
        "Pandemic and digital-learning conditions",
        "Ldisfunk01a_D",
        rationale="Contact with students supports participation and continuity.",
    ),
    F(
        "Distanzlernen – Deutschlehrkraft gab Rückmeldung zu Aufgaben",
        "Pandemic and digital-learning conditions",
        "Ldisfunk01d_D",
        rationale="Feedback supports correction, guidance and learning progress.",
    ),
)


if len(PREDICTOR_SPECS) != TOTAL_ALL_PREDICTORS:
    raise RuntimeError(
        f"Internal configuration error: expected {TOTAL_ALL_PREDICTORS} predictors, "
        f"found {len(PREDICTOR_SPECS)}."
    )

if len({spec.output_name for spec in PREDICTOR_SPECS}) != len(PREDICTOR_SPECS):
    raise RuntimeError("Internal configuration error: duplicate predictor names.")


# ---------------------------------------------------------------------------
# Metadata and input helpers
# ---------------------------------------------------------------------------


def target_spec(target_score: str = "pv-mean") -> FeatureSpec:
    if target_score != "pv-mean":
        raise ValueError("Version 8 verwendet ausschließlich den Mittelwert der 15 Plausible Values.")
    return TARGET_PV_MEAN


def selected_predictor_specs(feature_set: str) -> tuple[FeatureSpec, ...]:
    if feature_set == "all":
        return PREDICTOR_SPECS
    if feature_set != "selected15":
        raise ValueError(f"Unbekannte Feature-Auswahl: {feature_set}")

    by_name = {spec.output_name: spec for spec in PREDICTOR_SPECS}
    missing = [
        name
        for name in SELECTED15_OUTPUT_NAMES
        if name not in by_name and SELECTED15_ALIASES.get(name) not in by_name
    ]
    if missing:
        raise RuntimeError(
            "Interner Konfigurationsfehler: ausgewählte Features fehlen: "
            + ", ".join(missing)
        )
    result: list[FeatureSpec] = []
    for name in SELECTED15_OUTPUT_NAMES:
        alias_source_name = SELECTED15_ALIASES.get(name)
        if alias_source_name is None:
            result.append(by_name[name])
            continue

        source_spec = by_name[alias_source_name]
        result.append(
            FeatureSpec(
                output_name=name,
                role=source_spec.role,
                category=source_spec.category,
                sources=source_spec.sources,
                transform=source_spec.transform,
                rationale=source_spec.rationale,
                min_valid=source_spec.min_valid,
                prefer_non_imputed=source_spec.prefer_non_imputed,
            )
        )
    result = tuple(result)
    if len(result) != TOTAL_SELECTED_PREDICTORS:
        raise RuntimeError(
            f"Erwartet wurden {TOTAL_SELECTED_PREDICTORS} ausgewählte Prädiktoren, "
            f"gefunden wurden {len(result)}."
        )
    return result


def all_specs(target_score: str, feature_set: str) -> tuple[FeatureSpec, ...]:
    return (target_spec(target_score),) + selected_predictor_specs(feature_set)


def output_column_count(feature_set: str) -> int:
    return len(all_specs("pv-mean", feature_set))


def required_source_columns(target_score: str, feature_set: str) -> list[str]:
    return required_source_columns_from_specs(all_specs(target_score, feature_set))


def required_source_columns_from_specs(specs: Sequence[FeatureSpec]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        for source in spec.sources:
            if source not in seen:
                seen.add(source)
                result.append(source)
    return result


IMPUTATION_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<number>[1-9]|1[0-5])$")
PROTECTED_REPLICATE_SOURCES = frozenset(TARGET_PV_MEAN.sources)


def non_imputed_source(source: str, available: set[str]) -> str:
    """Prefer the original variable over an imputed replicate.

    IQB imputation families commonly use names such as ``Sind_1`` through
    ``Sind_15`` while the observed/original variable is named ``Sind``. A
    suffixed variable is replaced only when the unsuffixed base actually exists
    in the SAV metadata. This avoids changing legitimate names whose suffix is
    part of the original variable name. The 15 reading Plausible Values remain
    protected because they form the requested target.
    """
    if source in PROTECTED_REPLICATE_SOURCES:
        return source
    match = IMPUTATION_SUFFIX_RE.match(source)
    if match:
        base = match.group("base")
        if base in available:
            return base
    return source


def resolve_non_imputed_specs(
    specs: Sequence[FeatureSpec],
    available: set[str],
) -> tuple[FeatureSpec, ...]:
    """Return feature specifications that use original, non-imputed sources."""
    resolved: list[FeatureSpec] = []
    for spec in specs:
        if spec.role == "target" or not spec.prefer_non_imputed:
            resolved.append(spec)
            continue
        sources = tuple(non_imputed_source(source, available) for source in spec.sources)
        resolved.append(
            FeatureSpec(
                output_name=spec.output_name,
                role=spec.role,
                category=spec.category,
                sources=sources,
                transform=spec.transform,
                rationale=spec.rationale,
                min_valid=spec.min_valid,
                prefer_non_imputed=spec.prefer_non_imputed,
            )
        )
    return tuple(resolved)


def complete_replicate_family_bases(available: set[str]) -> set[str]:
    """Bases for which all numbered variants ``_1`` through ``_15`` exist."""
    members: dict[str, set[int]] = {}
    for column in available:
        match = IMPUTATION_SUFFIX_RE.match(column)
        if match:
            members.setdefault(match.group("base"), set()).add(
                int(match.group("number"))
            )
    expected = set(range(1, 16))
    return {base for base, numbers in members.items() if numbers == expected}


def is_imputed_duplicate(
    column: str,
    available: set[str],
    complete_families: set[str] | None = None,
) -> bool:
    """Whether a column belongs to an IQB 15-fold imputation/replicate family."""
    match = IMPUTATION_SUFFIX_RE.match(column)
    if not match:
        return False
    base = match.group("base")
    families = (
        complete_families
        if complete_families is not None
        else complete_replicate_family_bases(available)
    )
    return base in available or base in families


def absolute_all_source_columns(metadata: Any) -> list[str]:
    """All SAV columns except imputed or replicated 15-fold copies."""
    columns = [str(name) for name in getattr(metadata, "column_names", [])]
    available = set(columns)
    families = complete_replicate_family_bases(available)
    return [
        column
        for column in columns
        if not is_imputed_duplicate(column, available, families)
    ]


def discover_principal_sav(
    student_sav: Path,
    explicit_path: Path | None,
) -> Path | None:
    """Find the principal questionnaire SAV used for school-level originals."""
    if explicit_path is not None:
        return explicit_path
    exact = student_sav.with_name(
        "IQB-BT_2021_principal_v1_SUF_Off-site_2606-02a.sav"
    )
    if exact.is_file():
        return exact
    candidates = sorted(student_sav.parent.glob("*principal*.sav"))
    return candidates[0] if len(candidates) == 1 else None


def load_principal_originals(
    principal_path: Path,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, Any]:
    if pyreadstat is None:
        raise RuntimeError("pyreadstat ist nicht installiert")
    if not principal_path.is_file():
        raise FileNotFoundError(f"Schulleitungs-SAV nicht gefunden: {principal_path}")
    usecols = [SCHOOL_JOIN_KEY, *columns]
    frame, metadata = pyreadstat.read_sav(
        principal_path,
        usecols=usecols,
        apply_value_formats=False,
        formats_as_category=False,
    )
    if frame[SCHOOL_JOIN_KEY].duplicated().any():
        duplicated = frame.loc[
            frame[SCHOOL_JOIN_KEY].duplicated(keep=False), SCHOOL_JOIN_KEY
        ].dropna()
        if not duplicated.empty:
            raise ValueError(
                "Die Schulleitungs-SAV enthält mehrfach vorkommende Schul-IDs; "
                "die Originalwerte können nicht eindeutig zugeordnet werden."
            )
    return frame.set_index(SCHOOL_JOIN_KEY), metadata


def attach_principal_originals(
    dataframe: pd.DataFrame,
    lookup: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    if not columns:
        return dataframe
    if SCHOOL_JOIN_KEY not in dataframe.columns:
        raise ValueError(
            f"Die Schüler-SAV enthält den Verknüpfungsschlüssel {SCHOOL_JOIN_KEY} nicht."
        )
    result = dataframe.copy()
    school_ids = result[SCHOOL_JOIN_KEY]
    for column in columns:
        result[column] = school_ids.map(lookup[column])
    return result


def parse_delimiter(value: str) -> str:
    aliases = {
        "comma": ",",
        "semicolon": ";",
        "tab": "\t",
        r"\t": "\t",
        "pipe": "|",
    }
    delimiter = aliases.get(value.lower(), value)
    if len(delimiter) != 1:
        raise argparse.ArgumentTypeError(
            "Delimiter must be one character or comma, semicolon, tab or pipe."
        )
    return delimiter


MISSING_LABEL_PATTERNS: tuple[str, ...] = (
    "auslassen",
    "unklare beantwortung",
    "kein fragebogen",
    "nicht kalkulierbar",
    "unbekannt",
    "nicht enthalten",
    "nicht zuzuordnen",
    "keine notenangabe",
    "keine angabe",
    "sysmis",
    "habe ich nicht gemacht",
    "auslandsabschluss",
)


def infer_special_missing_codes(
    metadata: Any,
    variables: Sequence[str],
) -> dict[str, set[float]]:
    """Infer IQB administrative/missing codes from SPSS value labels."""
    value_labels: Mapping[str, Mapping[Any, Any]] = (
        getattr(metadata, "variable_value_labels", {}) or {}
    )
    result: dict[str, set[float]] = {}

    for variable in variables:
        codes: set[float] = set()
        labels = value_labels.get(variable, {}) or {}
        for raw_code, raw_label in labels.items():
            try:
                code = float(raw_code)
            except (TypeError, ValueError):
                continue
            label = " ".join(str(raw_label or "").casefold().split())
            if code < 0 or any(pattern in label for pattern in MISSING_LABEL_PATTERNS):
                codes.add(code)
        result[variable] = codes
    return result


def clean_numeric_sources(
    dataframe: pd.DataFrame,
    variables: Sequence[str],
    missing_codes: Mapping[str, set[float]],
) -> pd.DataFrame:
    """Coerce source columns to numeric and replace IQB missing codes with NA."""
    columns: dict[str, pd.Series] = {}
    for variable in variables:
        series = pd.to_numeric(dataframe[variable], errors="coerce")
        codes = missing_codes.get(variable, set())
        if codes:
            series = series.mask(series.isin(codes))
        columns[variable] = series
    return pd.DataFrame(columns, index=dataframe.index)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def row_mean_complete(frame: pd.DataFrame) -> pd.Series:
    """Compute a row mean only when all contributing source values are present."""
    valid_count = frame.notna().sum(axis=1)
    required_count = frame.shape[1]
    return frame.mean(axis=1, skipna=True).where(valid_count >= required_count)


def transform_feature(spec: FeatureSpec, source: pd.DataFrame) -> pd.Series:
    values = source.loc[:, list(spec.sources)]
    first = values.iloc[:, 0]

    if spec.transform == "raw":
        return first

    if spec.transform == "pv_mean":
        return row_mean_complete(values)

    if spec.transform == "mean":
        return row_mean_complete(values)

    if spec.transform == "grade_higher_better":
        valid = first.where(first.between(1, 6))
        return 7 - valid

    if spec.transform == "gym_recommendation":
        return first.map({1.0: 1.0, 2.0: 0.0})

    if spec.transform == "special_school_binary":
        return first.map({1.0: 0.0, 8.0: 1.0})

    if spec.transform == "home_german_higher":
        # Original recoded variable: 1=always German, 2=mixed, 3=never German.
        return first.map({1.0: 3.0, 2.0: 2.0, 3.0: 1.0})

    if spec.transform == "german_learning_age":
        # Original: 1=0-3, 2=4-5, 3=6-10, 9=not learned.
        return first.map({1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 9.0: 4.0})

    if spec.transform == "migration_proximity":
        # 0 none, 4 third generation, 1 one parent abroad,
        # 2 second generation, 3 first generation.
        return first.map({0.0: 0.0, 4.0: 1.0, 1.0: 2.0, 2.0: 3.0, 3.0: 4.0})

    if spec.transform == "any_positive":
        valid = first.where(first >= 0)
        result = (valid > 0).astype(float)
        return result.where(valid.notna())

    if spec.transform == "reverse_1_to_6":
        valid = first.where(first.between(1, 6))
        return 7 - valid

    raise ValueError(f"Unknown transformation: {spec.transform}")


def build_output_frame_from_specs(
    dataframe: pd.DataFrame,
    missing_codes: Mapping[str, set[float]],
    specs: Sequence[FeatureSpec],
) -> pd.DataFrame:
    required = required_source_columns_from_specs(specs)
    cleaned = clean_numeric_sources(dataframe, required, missing_codes)

    output = pd.DataFrame(
        {spec.output_name: transform_feature(spec, cleaned) for spec in specs},
        index=cleaned.index,
    )

    expected_names = [spec.output_name for spec in specs]
    if list(output.columns) != expected_names:
        raise RuntimeError(
            "Die Reihenfolge der Ausgabespalten stimmt nicht mit der Spezifikation überein."
        )
    if output.shape[1] != len(specs):
        raise RuntimeError(
            f"Erwartet wurden {len(specs)} Ausgabespalten, erhalten wurden {output.shape[1]}."
        )
    return output


def build_output_frame(
    dataframe: pd.DataFrame,
    missing_codes: Mapping[str, set[float]],
    target_score: str = "pv-mean",
    feature_set: str = "all",
) -> pd.DataFrame:
    """Compatibility helper using the unresolved static specification."""
    return build_output_frame_from_specs(
        dataframe, missing_codes, all_specs(target_score, feature_set)
    )


# ---------------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------------


def readable_transform(spec: FeatureSpec) -> str:
    descriptions = {
        "raw": "Numeric source value; IQB special missing codes replaced with NA and blanks preserved for separate missing-value visualization",
        "pv_mean": "Arithmetischer Mittelwert aus 15 Plausible Values; alle 15 Werte müssen gültig sein",
        "mean": "Mean across item battery; all source items required (no imputation)",
        "grade_higher_better": "Original grade recoded as 7 - grade, so higher values indicate better achievement",
        "gym_recommendation": "Gymnasium recommendation recoded to 1=yes and 0=no",
        "special_school_binary": "General school=0 and special school=1",
        "home_german_higher": "Reversed so 1=never German, 2=mixed and 3=always German",
        "german_learning_age": "0-3 years=1, 4-5=2, 6-10=3 and not learned=4",
        "migration_proximity": "None=0, third generation=1, one parent abroad=2, second generation=3, first generation=4",
        "any_positive": "Zero retained as 0; every positive category recoded to 1",
        "reverse_1_to_6": "Original 1-6 scale reversed, so higher values indicate more advantage",
    }
    return descriptions.get(spec.transform, spec.transform)


def write_feature_dictionary(
    path: Path,
    metadata: Any,
    target_score: str,
    feature_set: str,
    *,
    delimiter: str,
    encoding: str,
    specs: Sequence[FeatureSpec] | None = None,
    extra_labels: Mapping[str, Any] | None = None,
) -> None:
    labels: dict[str, Any] = dict(
        getattr(metadata, "column_names_to_labels", {}) or {}
    )
    if extra_labels:
        labels.update(extra_labels)
    rows: list[dict[str, Any]] = []

    role_names = {"target": "Zielvariable", "predictor": "Prädiktor"}
    active_specs = tuple(specs) if specs is not None else all_specs(target_score, feature_set)
    for rank, spec in enumerate(active_specs, start=1):
        source_labels = list(
            dict.fromkeys(str(labels.get(source, "") or "") for source in spec.sources)
        )
        rows.append(
            {
                "rank": rank,
                "role": role_names.get(spec.role, spec.role),
                "category": spec.category,
                "output_column": spec.output_name,
                "source_variables": " | ".join(spec.sources),
                "source_labels": " | ".join(source_labels),
                "transformation": readable_transform(spec),
                "selection_rationale": spec.rationale,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        sep=delimiter,
        encoding=encoding,
    )


# ---------------------------------------------------------------------------
# Absolute-all export helpers
# ---------------------------------------------------------------------------


def compact_label(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def absolute_all_headers(
    metadata: Any,
    columns: Sequence[str],
) -> dict[str, str]:
    """Create readable, unique headers while preserving the SPSS code."""
    labels: Mapping[str, Any] = getattr(metadata, "column_names_to_labels", {}) or {}
    result: dict[str, str] = {}
    used: set[str] = {TARGET_PV_MEAN.output_name}
    for column in columns:
        label = compact_label(labels.get(column, ""))
        candidate = f"{label} [{column}]" if label else column
        if candidate in used:
            candidate = f"{candidate} ({column})"
        counter = 2
        base_candidate = candidate
        while candidate in used:
            candidate = f"{base_candidate} ({counter})"
            counter += 1
        used.add(candidate)
        result[column] = candidate
    return result


def clean_absolute_all_frame(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    missing_codes: Mapping[str, set[float]],
) -> pd.DataFrame:
    """Clean IQB missing codes but otherwise preserve original SAV values."""
    output: dict[str, pd.Series] = {}
    for column in columns:
        series = dataframe[column].copy()
        codes = missing_codes.get(column, set())
        if codes:
            numeric = pd.to_numeric(series, errors="coerce")
            # Apply numeric missing codes only where conversion is meaningful.
            mask = numeric.isin(codes)
            if mask.any():
                series = series.mask(mask)
        output[column] = series
    return pd.DataFrame(output, index=dataframe.index)


def build_absolute_all_output_frame(
    dataframe: pd.DataFrame,
    source_columns: Sequence[str],
    headers: Mapping[str, str],
    missing_codes: Mapping[str, set[float]],
) -> pd.DataFrame:
    target_required = list(TARGET_PV_MEAN.sources)
    target_clean = clean_numeric_sources(dataframe, target_required, missing_codes)
    target = transform_feature(TARGET_PV_MEAN, target_clean)
    raw = clean_absolute_all_frame(dataframe, source_columns, missing_codes)
    raw = raw.rename(columns=dict(headers))
    return pd.concat(
        [target.rename(TARGET_PV_MEAN.output_name), raw],
        axis=1,
    )


def write_absolute_all_dictionary(
    path: Path,
    metadata: Any,
    columns: Sequence[str],
    headers: Mapping[str, str],
    *,
    delimiter: str,
    encoding: str,
) -> None:
    labels: Mapping[str, Any] = getattr(metadata, "column_names_to_labels", {}) or {}
    rows: list[dict[str, Any]] = [
        {
            "rank": 1,
            "role": "Zielvariable",
            "output_column": TARGET_PV_MEAN.output_name,
            "source_variable": " | ".join(TARGET_PV_MEAN.sources),
            "source_label": "",
            "transformation": readable_transform(TARGET_PV_MEAN),
            "non_imputed_original": "n/a – Plausible Values form the target",
        }
    ]
    for rank, column in enumerate(columns, start=2):
        rows.append(
            {
                "rank": rank,
                "role": "Originalvariable",
                "output_column": headers[column],
                "source_variable": column,
                "source_label": compact_label(labels.get(column, "")),
                "transformation": "Originalwert; IQB-Sonderfehlwerte werden leer ausgegeben",
                "non_imputed_original": "ja",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        sep=delimiter,
        encoding=encoding,
    )


# ---------------------------------------------------------------------------
# Random participant sampling
# ---------------------------------------------------------------------------


def build_sample_indices(
    total_rows: int,
    *,
    sample_fraction: float,
    random_seed: int,
) -> np.ndarray:
    """Return sorted zero-based row positions for an exact random sample.

    Sampling is performed without replacement. ``round`` is used because a
    fraction such as 15 percent will usually not produce an integer number of
    rows. The same total row count and seed always yield the same sample.
    """
    if total_rows < 0:
        raise ValueError("Die Gesamtzahl der Zeilen darf nicht negativ sein")
    if not 0 < sample_fraction <= 1:
        raise ValueError("Der interne Stichprobenanteil muss größer als 0 und höchstens 1 sein")
    if total_rows == 0:
        return np.empty(0, dtype=np.int64)

    sample_size = int(round(total_rows * sample_fraction))
    sample_size = max(1, min(total_rows, sample_size))
    generator = np.random.default_rng(random_seed)
    return np.sort(
        generator.choice(total_rows, size=sample_size, replace=False).astype(np.int64)
    )


def sample_chunk_by_global_position(
    dataframe: pd.DataFrame,
    *,
    chunk_start: int,
    sample_indices: np.ndarray,
) -> pd.DataFrame:
    """Keep sampled rows from one chunk using positions in the full SAV file."""
    chunk_end = chunk_start + len(dataframe)
    left = int(np.searchsorted(sample_indices, chunk_start, side="left"))
    right = int(np.searchsorted(sample_indices, chunk_end, side="left"))
    if left == right:
        return dataframe.iloc[0:0].copy()

    local_positions = sample_indices[left:right] - chunk_start
    return dataframe.iloc[local_positions].copy()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_sav_to_csv(
    input_path: Path,
    output_path: Path,
    dictionary_path: Path,
    *,
    target_score: str,
    feature_set: str,
    chunksize: int,
    delimiter: str,
    encoding: str,
    limit: int,
    sample_fraction: float,
    random_seed: int,
    principal_path: Path | None = None,
) -> None:
    if pyreadstat is None:
        raise RuntimeError(
            "Die Abhängigkeit 'pyreadstat' fehlt. Installation:\n"
            "  python -m pip install pandas numpy pyreadstat"
        )
    if feature_set not in {"selected15", "all", "absolute-all"}:
        raise ValueError(
            "feature_set muss 'selected15', 'all' oder 'absolute-all' sein"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_path}")
    if chunksize <= 0:
        raise ValueError("--chunksize muss größer als null sein")
    if limit < 0:
        raise ValueError("--limit darf nicht negativ sein")
    if not 0 < sample_fraction <= 1:
        raise ValueError(
            "Der interne Stichprobenanteil muss größer als 0 und höchstens 1 sein"
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Eingabe- und Ausgabepfad müssen unterschiedlich sein")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")

    _, metadata = pyreadstat.read_sav(input_path, metadataonly=True)
    total_rows_raw = getattr(metadata, "number_rows", None)
    if total_rows_raw is None:
        raise RuntimeError(
            "Die Gesamtzahl der SAV-Zeilen konnte nicht aus den Metadaten gelesen werden."
        )
    total_rows = int(total_rows_raw)
    sample_indices = build_sample_indices(
        total_rows,
        sample_fraction=sample_fraction,
        random_seed=random_seed,
    )
    sample_size = len(sample_indices)

    metadata_columns = [str(name) for name in metadata.column_names]
    available = set(metadata_columns)

    resolved_specs: tuple[FeatureSpec, ...] | None = None
    absolute_columns: list[str] | None = None
    absolute_headers_map: dict[str, str] | None = None
    replacement_pairs: list[tuple[str, str]] = []
    excluded_imputation_count = 0
    principal_lookup: pd.DataFrame | None = None
    principal_metadata: Any | None = None
    principal_used_path: Path | None = None
    external_required: list[str] = []
    extra_labels: dict[str, Any] = {}

    if feature_set == "absolute-all":
        absolute_columns = absolute_all_source_columns(metadata)
        excluded_imputation_count = len(metadata_columns) - len(absolute_columns)
        absolute_headers_map = absolute_all_headers(metadata, absolute_columns)
        required = list(dict.fromkeys([*TARGET_PV_MEAN.sources, *absolute_columns]))
        student_required = required
        expected_columns = 1 + len(absolute_columns)
        output_headers = [
            TARGET_PV_MEAN.output_name,
            *(absolute_headers_map[column] for column in absolute_columns),
        ]
    else:
        static_specs = all_specs(target_score, feature_set)

        # Two school-context variables exist without imputation only in the
        # separate school-principal SAV. Load its metadata when available so
        # that Pdisbt_a_sum_1/Pdisbt_b_sum_1 can become Pdisbt_a_sum/Pdisbt_b_sum.
        principal_candidate = discover_principal_sav(input_path, principal_path)
        principal_available: set[str] = set()
        if principal_candidate is not None and principal_candidate.is_file():
            _, principal_metadata = pyreadstat.read_sav(
                principal_candidate, metadataonly=True
            )
            principal_available = {
                str(name) for name in principal_metadata.column_names
            }
            principal_used_path = principal_candidate

        resolution_available = available | principal_available
        resolved_specs = resolve_non_imputed_specs(
            static_specs, resolution_available
        )
        required = required_source_columns_from_specs(resolved_specs)
        external_required = [source for source in required if source not in available]

        missing_combined = sorted(
            set(required) - available - principal_available
        )
        if missing_combined:
            preview = ", ".join(missing_combined[:20])
            suffix = " ..." if len(missing_combined) > 20 else ""
            raise ValueError(
                f"Den verfügbaren SAV-Dateien fehlen {len(missing_combined)} "
                f"benötigte Originalvariablen: {preview}{suffix}."
            )

        if external_required:
            if principal_used_path is None:
                raise ValueError(
                    "Für die nicht-imputierten Schulkontextvariablen wird die "
                    "Schulleitungs-SAV benötigt. Lege die Datei "
                    "IQB-BT_2021_principal_v1_SUF_Off-site_2606-02a.sav in "
                    "denselben Ordner oder übergib sie mit --principal-sav."
                )
            principal_lookup, principal_metadata = load_principal_originals(
                principal_used_path,
                external_required,
            )
            extra_labels = dict(
                getattr(principal_metadata, "column_names_to_labels", {}) or {}
            )

        student_required = [source for source in required if source in available]
        if external_required and SCHOOL_JOIN_KEY not in student_required:
            student_required.append(SCHOOL_JOIN_KEY)

        expected_columns = len(resolved_specs)
        output_headers = [spec.output_name for spec in resolved_specs]
        for static_spec, resolved_spec in zip(static_specs, resolved_specs):
            for old, new_source in zip(static_spec.sources, resolved_spec.sources):
                if old != new_source:
                    replacement_pairs.append((old, new_source))

    missing_student = sorted(set(student_required) - available)
    if missing_student:
        preview = ", ".join(missing_student[:20])
        suffix = " ..." if len(missing_student) > 20 else ""
        raise ValueError(
            f"Der Schüler-SAV fehlen {len(missing_student)} benötigte Variablen: "
            f"{preview}{suffix}."
        )

    missing_codes = infer_special_missing_codes(metadata, student_required)
    if external_required:
        assert principal_metadata is not None
        missing_codes.update(
            infer_special_missing_codes(principal_metadata, external_required)
        )

    if feature_set == "absolute-all":
        assert absolute_columns is not None
        assert absolute_headers_map is not None
        write_absolute_all_dictionary(
            dictionary_path,
            metadata,
            absolute_columns,
            absolute_headers_map,
            delimiter=delimiter,
            encoding=encoding,
        )
    else:
        assert resolved_specs is not None
        write_feature_dictionary(
            dictionary_path,
            metadata,
            target_score,
            feature_set,
            delimiter=delimiter,
            encoding=encoding,
            specs=resolved_specs,
            extra_labels=extra_labels,
        )

    print(
        f"sav_to_csv_v8_de_random_percent.py {SCRIPT_VERSION}: {input_path.name}",
        file=sys.stderr,
    )
    print(f"Feature-Auswahl: {feature_set}", file=sys.stderr)
    if feature_set == "absolute-all":
        print(
            f"Ausgabe: {expected_columns:,} Spalten "
            f"(1 Zielvariable + {expected_columns - 1:,} nicht-imputierte Originalvariablen).",
            file=sys.stderr,
        )
        print(
            f"Entfernte imputierte Kopien: {excluded_imputation_count:,}.",
            file=sys.stderr,
        )
    else:
        print(
            f"Ausgabe: {expected_columns} Spalten "
            f"(1 Zielvariable + {expected_columns - 1} Prädiktoren).",
            file=sys.stderr,
        )
        if replacement_pairs:
            print(
                f"Durch Originalvariablen ersetzte imputierte Quellen: "
                f"{len(set(replacement_pairs))}.",
                file=sys.stderr,
            )
            for imputed, original in sorted(set(replacement_pairs))[:15]:
                print(f"  {imputed} -> {original}", file=sys.stderr)
            if len(set(replacement_pairs)) > 15:
                print("  ...", file=sys.stderr)
        else:
            print(
                "Kuratierten Prädiktoren verwenden nicht-imputierte "
                "Originalvariablen.",
                file=sys.stderr,
            )
        if external_required and principal_used_path is not None:
            print(
                f"Schulkontext-Originalwerte aus: {principal_used_path.name}",
                file=sys.stderr,
            )

    print(
        "Personenfilter: keiner – Auswahl ausschließlich per Zufallsstichprobe.",
        file=sys.stderr,
    )
    print(
        f"Zufallsstichprobe: {sample_size:,} von {total_rows:,} Zeilen "
        f"({sample_size / total_rows:.2%}; Seed {random_seed})."
        if total_rows
        else f"Zufallsstichprobe: 0 von 0 Zeilen; Seed {random_seed}.",
        file=sys.stderr,
    )

    retained_rows = 0
    scanned_rows = 0
    first_written_chunk = True

    try:
        iterator = pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav,
            input_path,
            chunksize=chunksize,
            usecols=student_required,
            apply_value_formats=False,
            formats_as_category=False,
        )

        for dataframe, _chunk_metadata in iterator:
            chunk_start = scanned_rows
            scanned_rows += len(dataframe)
            sampled_dataframe = sample_chunk_by_global_position(
                dataframe,
                chunk_start=chunk_start,
                sample_indices=sample_indices,
            )
            if external_required:
                assert principal_lookup is not None
                sampled_dataframe = attach_principal_originals(
                    sampled_dataframe,
                    principal_lookup,
                    external_required,
                )

            if feature_set == "absolute-all":
                assert absolute_columns is not None
                assert absolute_headers_map is not None
                output = build_absolute_all_output_frame(
                    sampled_dataframe,
                    absolute_columns,
                    absolute_headers_map,
                    missing_codes,
                )
            else:
                assert resolved_specs is not None
                output = build_output_frame_from_specs(
                    sampled_dataframe,
                    missing_codes,
                    resolved_specs,
                )

            if limit:
                remaining = limit - retained_rows
                if remaining <= 0:
                    break
                if len(output) > remaining:
                    output = output.iloc[:remaining].copy()

            if not output.empty:
                output.to_csv(
                    temporary_path,
                    mode="w" if first_written_chunk else "a",
                    header=first_written_chunk,
                    index=False,
                    sep=delimiter,
                    encoding=encoding,
                    na_rep="",
                )
                retained_rows += len(output)
                first_written_chunk = False

            print(
                f"  {scanned_rows:,} Zeilen gelesen; {retained_rows:,} geschrieben",
                file=sys.stderr,
            )

            if limit and retained_rows >= limit:
                break

        if first_written_chunk:
            pd.DataFrame(columns=output_headers).to_csv(
                temporary_path,
                index=False,
                sep=delimiter,
                encoding=encoding,
            )

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Fertig: {output_path} ({retained_rows:,} Zeilen, "
        f"{expected_columns:,} Spalten, {size_mb:,.1f} MB)",
        file=sys.stderr,
    )
    print(f"Feature-Legende: {dictionary_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def format_percent_tag(sample_percent: float) -> str:
    """Return a filesystem-friendly percentage label, e.g. 12.5 -> 12p5pct."""
    normalized = f"{sample_percent:.10f}".rstrip("0").rstrip(".")
    return normalized.replace(".", "p") + "pct"


def default_output_path(
    input_path: Path,
    feature_set: str,
    sample_percent: float,
) -> Path:
    parts = {
        "selected15": "selected15",
        "all": "top100",
        "absolute-all": "absolute_all_non_imputed",
    }
    part = parts[feature_set]
    percent_tag = format_percent_tag(sample_percent)
    return input_path.with_name(
        f"{input_path.stem}_german_reading_{part}_v8_de_random{percent_tag}_pv_mean.csv"
    )


def default_dictionary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_feature_dictionary.csv")


def print_feature_list(feature_set: str, target_score: str) -> None:
    sets = ("selected15", "all") if feature_set == "both" else (feature_set,)
    for current in sets:
        if current == "absolute-all":
            print(
                "\n[absolute-all] Dynamische Ausgabe: Zielvariable plus alle "
                "Originalvariablen der SAV-Datei. Vollständige 15-fache "
                "Imputationsfamilien X_1 bis X_15 werden ausgelassen."
            )
            continue
        specs = all_specs(target_score, current)
        print(
            f"\n[{current}] {len(specs)} Ausgabespalten; keine Personenfilterung; "
            "Originalquellen werden bevorzugt, außer Ssoe/Sind/Sdeangst mit _1; "
            "Stichprobenanteil wird mit --sample-percent festgelegt"
        )
        for rank, spec in enumerate(specs, start=1):
            role = "Zielvariable" if spec.role == "target" else "Prädiktor"
            print(f"{rank:03d}. [{role}] {spec.output_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Erstellt einen deutschen IQB-Lesekompetenz-Datensatz ohne "
            "personenbezogene Ausschlüsse und zieht eine reproduzierbare "
            "Zufallsstichprobe mit frei wählbarem Prozentanteil. Die Zielvariable "
            "ist der Mittelwert der 15 Plausible Values. Für Prädiktoren werden "
            "grundsätzlich Originalvariablen statt imputierter Kopien verwendet; "
            "Ssoe, Sind und Sdeangst bleiben auf den imputierten _1-Variablen."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Pfad zur IQB-BT_2021_student_quest...sav",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help=(
            "Optionaler Ausgabepfad. Nicht zusammen mit --feature-set both verwendbar."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=("selected15", "all", "both", "absolute-all"),
        default="all",
        help=(
            "selected15 = Ziel + 14 ausgewählte Features; all = Ziel + 99 Features; "
            "both = beide kuratierten Dateien; absolute-all = Ziel + alle "
            "nicht-imputierten Originalvariablen (Standard: all)"
        ),
    )
    parser.add_argument(
        "--dictionary-output",
        type=Path,
        help=(
            "Optionaler Pfad der Feature-Legende. Nicht zusammen mit "
            "--feature-set both verwendbar."
        ),
    )
    parser.add_argument(
        "--target-score",
        choices=("pv-mean",),
        default="pv-mean",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--principal-sav",
        type=Path,
        help=(
            "Optionaler Pfad zur IQB-Schulleitungs-SAV. Sie wird für die "
            "nicht-imputierten Originalwerte Pdisbt_a_sum und Pdisbt_b_sum "
            "benötigt und sonst automatisch im Ordner der Schüler-SAV gesucht."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2_000,
        help="Pro Block gelesene Zeilen (Standard: 2000)",
    )
    parser.add_argument(
        "--delimiter",
        type=parse_delimiter,
        default=",",
        help="CSV-Trennzeichen: comma, semicolon, tab, pipe oder ein Zeichen",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Ausgabe-Encoding (Standard: utf-8-sig, Excel-kompatibel)",
    )
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=None,
        metavar="PROZENT",
        help=(
            "Prozentanteil der zufällig zu exportierenden Teilnehmenden, "
            "zum Beispiel 15 für 15 Prozent. Erforderlich beim Export."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Zufalls-Seed für eine reproduzierbare Stichprobe (Standard: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximal N Zeilen exportieren; 0 = alle",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Ausgabefeatures anzeigen und beenden",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_features:
        print_feature_list(args.feature_set, args.target_score)
        return 0

    if args.input is None:
        parser.error("Die Eingabe-SAV ist erforderlich, außer bei --list-features.")

    if args.sample_percent is None:
        parser.error("--sample-percent ist beim Export erforderlich, z. B. --sample-percent 15.")
    if not 0 < args.sample_percent <= 100:
        parser.error("--sample-percent muss größer als 0 und höchstens 100 sein.")

    sample_fraction = args.sample_percent / 100.0

    if args.feature_set == "both" and (args.output or args.dictionary_output):
        parser.error(
            "Bei --feature-set both dürfen output und --dictionary-output nicht "
            "angegeben werden, da zwei Dateien erzeugt werden."
        )

    feature_sets = (
        ("selected15", "all") if args.feature_set == "both" else (args.feature_set,)
    )

    try:
        for current in feature_sets:
            output_path = args.output or default_output_path(
                args.input,
                current,
                args.sample_percent,
            )
            dictionary_path = (
                args.dictionary_output or default_dictionary_path(output_path)
            )
            convert_sav_to_csv(
                args.input,
                output_path,
                dictionary_path,
                target_score=args.target_score,
                feature_set=current,
                chunksize=args.chunksize,
                delimiter=args.delimiter,
                encoding=args.encoding,
                limit=args.limit,
                sample_fraction=sample_fraction,
                random_seed=args.random_seed,
                principal_path=args.principal_sav,
            )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    except Exception as exc:
        print(f"Konvertierung fehlgeschlagen: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
