"""Dataset upload, feature provenance, and questionnaire metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import json
import re

import numpy as np
import pandas as pd

from .ml_shared import (
    CASE_STUDY_FEATURE_NAMES,
    CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN,
    CASE_STUDY_TARGET_COLUMN,
    MAX_PREVIEW_ROWS,
    _json_safe,
    _resolve_fallback_data_file,
)


class DatasetProvenanceMixin:
    def _restore_active_dataset_metadata(self) -> None:
        if not self.active_dataset_file.exists():
            return
        try:
            data = json.loads(self.active_dataset_file.read_text(encoding="utf-8"))
        except Exception:
            return

        dataset_path = data.get("dataset_path")
        target_column = data.get("target_column")
        selected_feature_columns = data.get("selected_feature_columns")
        dataset_id = data.get("dataset_id")
        dataset_name = data.get("dataset_name")
        feature_chart_settings = data.get("feature_chart_settings")
        show_missing_bars = data.get("show_missing_bars")
        show_competence_levels = data.get("show_competence_levels")
        show_prediction_comparison_superadmin = data.get(
            "show_prediction_comparison_superadmin"
        )
        show_prediction_comparison_users = data.get(
            "show_prediction_comparison_users"
        )
        if dataset_path:
            self.active_dataset_path = str(Path(dataset_path))
        if target_column:
            self.target_column = str(target_column)
        if isinstance(selected_feature_columns, list):
            self.selected_feature_columns = [str(col) for col in selected_feature_columns]
        if dataset_id:
            self.active_dataset_id = str(dataset_id)
        if dataset_name:
            self.active_dataset_name = str(Path(str(dataset_name)).name)
        if isinstance(feature_chart_settings, dict):
            sanitized: Dict[str, Dict[str, Any]] = {}
            for feature_name, raw_setting in feature_chart_settings.items():
                normalized = self._normalize_stored_feature_chart_setting(raw_setting)
                if normalized:
                    sanitized[str(feature_name)] = normalized
            self.feature_chart_settings = sanitized
        if show_missing_bars is not None:
            self.show_missing_bars = bool(show_missing_bars)
        if show_competence_levels is not None:
            self.show_competence_levels = bool(show_competence_levels)
        if show_prediction_comparison_superadmin is not None:
            self.show_prediction_comparison_superadmin = bool(
                show_prediction_comparison_superadmin
            )
        if show_prediction_comparison_users is not None:
            self.show_prediction_comparison_users = bool(show_prediction_comparison_users)

    def _persist_active_dataset_metadata(self) -> None:
        payload = {
            "dataset_id": self.active_dataset_id,
            "dataset_path": self.active_dataset_path,
            "dataset_name": self.active_dataset_name,
            "target_column": self.target_column,
            "selected_feature_columns": self.selected_feature_columns,
            "feature_chart_settings": self.feature_chart_settings,
            "show_missing_bars": bool(self.show_missing_bars),
            "show_competence_levels": bool(self.show_competence_levels),
            "show_prediction_comparison_superadmin": bool(
                self.show_prediction_comparison_superadmin
            ),
            "show_prediction_comparison_users": bool(
                self.show_prediction_comparison_users
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.active_dataset_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _auto_load_persisted_dataset(self) -> None:
        if not self.active_dataset_path or not self.target_column:
            return
        dataset_path = Path(self.active_dataset_path)
        if not dataset_path.is_file():
            return
        try:
            self.load_data(
                dataset_id=self.active_dataset_id,
                target_column=self.target_column,
                feature_columns=self.selected_feature_columns or None,
            )
        except Exception:
            # Keep service usable even if persisted dataset is invalid.
            self._reset_runtime_state()

    def _resolve_dataset_path(self, dataset_id: Optional[str] = None) -> Tuple[Path, Optional[str]]:
        if dataset_id:
            safe_name = Path(dataset_id).name
            candidate = (self.datasets_dir / safe_name).resolve()
            if not str(candidate).startswith(str(self.datasets_dir.resolve())):
                raise ValueError("Invalid dataset_id")
            if not candidate.is_file():
                raise FileNotFoundError(f"Uploaded dataset not found: {safe_name}")
            return candidate, safe_name

        if self.active_dataset_path:
            active_path = Path(self.active_dataset_path).resolve()
            if active_path.is_file():
                return active_path, self.active_dataset_id

        return _resolve_fallback_data_file(), None

    def inspect_uploaded_dataset(
        self,
        file_name: str,
        file_bytes: bytes,
        *,
        use_case_study_feature_names: bool = False,
    ) -> Dict[str, Any]:
        """Persist an uploaded CSV and return column preview information."""
        if not file_name or not file_name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported")
        if not file_bytes:
            raise ValueError("Uploaded file is empty")

        dataset_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex}.csv"
        dataset_path = self.datasets_dir / dataset_id
        dataset_path.write_bytes(file_bytes)

        try:
            if use_case_study_feature_names:
                case_study_df = pd.read_csv(dataset_path)
                renamed_columns = {
                    source_name: exported_name
                    for source_name, exported_name in (
                        CASE_STUDY_FEATURE_NAME_BY_SOURCE_COLUMN.items()
                    )
                    if source_name in case_study_df.columns
                }
                if renamed_columns:
                    case_study_df = case_study_df.rename(columns=renamed_columns)
                    case_study_df.to_csv(dataset_path, index=False)
            preview_df = pd.read_csv(dataset_path, nrows=MAX_PREVIEW_ROWS)
        except Exception as exc:
            dataset_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

        columns = [str(col) for col in preview_df.columns]
        if len(columns) < 2:
            dataset_path.unlink(missing_ok=True)
            raise ValueError("CSV must contain at least one target and one feature column")

        default_target = (
            CASE_STUDY_TARGET_COLUMN
            if use_case_study_feature_names and CASE_STUDY_TARGET_COLUMN in columns
            else ("cnt" if "cnt" in columns else columns[-1])
        )
        return {
            "dataset_id": dataset_id,
            "original_filename": file_name,
            "columns": columns,
            "default_target_column": default_target,
            "case_study_feature_columns": (
                [name for name in CASE_STUDY_FEATURE_NAMES if name in columns]
                if use_case_study_feature_names
                else None
            ),
        }

    @staticmethod
    def _split_dictionary_values(raw_value: Any) -> List[str]:
        if raw_value is None:
            return []
        text = str(raw_value).strip()
        if not text or text.lower() == "nan":
            return []
        return [part.strip() for part in text.split("|") if part.strip()]

    @staticmethod
    def _classify_feature_construction(
        transformation: str,
        source_labels: List[str],
    ) -> str:
        normalized_transformation = transformation.strip().lower()
        normalized_labels = " ".join(source_labels).strip().lower()

        if (
            "mean across item battery" in normalized_transformation
            or "itemmittelwert" in normalized_transformation
            or "mittelwert aus" in normalized_transformation
            or "row mean across item battery" in normalized_transformation
        ):
            return "item_mean"

        if "scale" in normalized_labels and "imputiert" in normalized_labels:
            return "iqb_scale"
        if "skala" in normalized_labels and "imputiert" in normalized_labels:
            return "iqb_scale"

        return "raw_source"

    @staticmethod
    def _extract_missing_value_handling(transformation: str) -> str:
        text = str(transformation or "").strip()
        if not text:
            return ""

        parts = [part.strip() for part in text.split(";") if part.strip()]
        if len(parts) > 1:
            return "; ".join(parts[1:])

        lower_text = text.lower()
        if "missing codes replaced with na" in lower_text:
            return "IQB special missing codes replaced with NA"
        if "sonderfehlwerte" in lower_text:
            return text

        return ""

    @staticmethod
    def _normalize_scale_variable_name(variable_name: Any) -> str:
        normalized = str(variable_name or "").strip()
        if not normalized:
            return ""
        return re.sub(r"_\d+$", "", normalized)

    @staticmethod
    def _normalize_scale_topic(scale_label: Any) -> str:
        normalized = str(scale_label or "").strip()
        if not normalized:
            return ""
        normalized = re.sub(r"^(skala|scale)\s*-\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s*\(imputiert\)\s*$", "", normalized, flags=re.IGNORECASE)
        return normalized.strip()

    @staticmethod
    def _shared_response_options_from_details(
        detail_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        common_options: Optional[List[Dict[str, Any]]] = None

        for detail in detail_rows:
            options = _json_safe(detail.get("response_options", []))
            if not options:
                continue

            if common_options is None:
                common_options = options
            elif common_options != options:
                return []

        return common_options or []

    @staticmethod
    def _scale_topic_tokens(scale_topic: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", str(scale_topic or "").lower())
            if token
        ]

    def _infer_iqb_scale_item_details(
        self,
        source_variables: List[str],
        source_labels: List[str],
        response_option_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        inferred_details: List[Dict[str, Any]] = []
        seen_variables: set[str] = set()

        def append_candidate(variable: str, metadata: Dict[str, Any]) -> None:
            if variable in seen_variables:
                return

            inferred_details.append(
                {
                    "variable": variable,
                    "label": str(metadata.get("description", "") or "").strip(),
                    "response_options": _json_safe(
                        metadata.get("response_options", [])
                    ),
                    "missing_response_options": _json_safe(
                        metadata.get("missing_response_options", [])
                    ),
                }
            )
            seen_variables.add(variable)

        for index, source_variable in enumerate(source_variables):
            base_variable = self._normalize_scale_variable_name(source_variable)
            if not base_variable:
                continue

            scale_topic = self._normalize_scale_topic(
                source_labels[index] if index < len(source_labels) else "",
            )
            scale_topic_lower = scale_topic.lower()
            scale_topic_tokens = self._scale_topic_tokens(scale_topic)

            for candidate_variable, metadata in response_option_map.items():
                if candidate_variable in seen_variables:
                    continue
                if candidate_variable == source_variable or candidate_variable == base_variable:
                    continue
                if re.fullmatch(rf"{re.escape(base_variable)}_\d+", candidate_variable):
                    continue
                if not candidate_variable.startswith(base_variable):
                    continue

                response_options = _json_safe(metadata.get("response_options", []))
                if not response_options:
                    continue

                description = str(metadata.get("description", "") or "").strip()
                if not description:
                    continue
                if description.lower().startswith(("skala -", "scale -")):
                    continue
                if scale_topic_lower and scale_topic_lower not in description.lower():
                    continue

                append_candidate(candidate_variable, metadata)

            if inferred_details or not scale_topic_tokens:
                continue

            for candidate_variable, metadata in response_option_map.items():
                if candidate_variable in seen_variables:
                    continue
                if candidate_variable == source_variable or candidate_variable == base_variable:
                    continue

                response_options = _json_safe(metadata.get("response_options", []))
                if not response_options:
                    continue

                description = str(metadata.get("description", "") or "").strip()
                description_lower = description.lower()
                if not description:
                    continue
                if description_lower.startswith(("skala -", "scale -")):
                    continue
                if not all(token in description_lower for token in scale_topic_tokens):
                    continue

                append_candidate(candidate_variable, metadata)

        inferred_details.sort(key=lambda detail: str(detail.get("variable", "")))
        return inferred_details

    def _load_feature_provenance_from_path(
        self,
        dictionary_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        cache_key = str(dictionary_path)
        cached = self._feature_dictionary_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            dictionary_df = pd.read_csv(dictionary_path, dtype=str, encoding="utf-8-sig")
        except Exception:
            self._feature_dictionary_cache[cache_key] = {}
            return {}

        required_columns = {
            "output_column",
            "role",
            "category",
            "source_variables",
            "source_labels",
            "transformation",
            "selection_rationale",
        }
        if not required_columns.issubset(set(dictionary_df.columns)):
            self._feature_dictionary_cache[cache_key] = {}
            return {}

        response_option_map = self._build_response_option_map_for_dictionary(
            dictionary_path,
        )
        provenance_map: Dict[str, Dict[str, Any]] = {}
        for _, row in dictionary_df.iterrows():
            output_column = str(row.get("output_column", "") or "").strip()
            if not output_column:
                continue

            source_variables = self._split_dictionary_values(row.get("source_variables"))
            source_labels = self._split_dictionary_values(row.get("source_labels"))
            transformation = str(row.get("transformation", "") or "").strip()
            selection_rationale = str(row.get("selection_rationale", "") or "").strip()
            source_details = []
            common_response_options: Optional[List[Dict[str, Any]]] = None
            construction_type = self._classify_feature_construction(
                transformation,
                source_labels,
            )

            for index, variable in enumerate(source_variables):
                response_metadata = response_option_map.get(variable, {})
                detail = {
                    "variable": variable,
                    "label": source_labels[index] if index < len(source_labels) else "",
                    "response_options": _json_safe(
                        response_metadata.get("response_options", [])
                    ),
                    "missing_response_options": _json_safe(
                        response_metadata.get("missing_response_options", [])
                    ),
                }
                source_details.append(detail)

                substantive_options = detail["response_options"]
                if not substantive_options:
                    common_response_options = None
                    continue

                if common_response_options is None:
                    common_response_options = substantive_options
                elif common_response_options != substantive_options:
                    common_response_options = []

            scale_item_details: List[Dict[str, Any]] = []
            if construction_type == "iqb_scale":
                scale_item_details = self._infer_iqb_scale_item_details(
                    source_variables,
                    source_labels,
                    response_option_map,
                )
                if not common_response_options:
                    common_response_options = self._shared_response_options_from_details(
                        scale_item_details,
                    )

            provenance_map[output_column] = {
                "role": str(row.get("role", "") or "").strip(),
                "category": str(row.get("category", "") or "").strip(),
                "source_variables": source_variables,
                "source_labels": source_labels,
                "source_details": source_details,
                "scale_item_details": scale_item_details,
                "transformation": transformation,
                "selection_rationale": selection_rationale,
                "construction_type": construction_type,
                "response_options": common_response_options or [],
                "missing_value_handling": self._extract_missing_value_handling(
                    transformation,
                ),
                "source_count": max(len(source_variables), len(source_labels)),
                "dictionary_path": str(dictionary_path),
            }

        self._feature_dictionary_cache[cache_key] = provenance_map
        return provenance_map

    def _find_feature_dictionary_path_by_feature_names(
        self,
        feature_names: List[str],
    ) -> Optional[Path]:
        normalized_feature_names = {
            str(feature_name).strip()
            for feature_name in feature_names
            if str(feature_name).strip()
        }
        if not normalized_feature_names:
            return None

        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [repo_root, Path.home() / "Downloads"]
        candidate_matches: List[Tuple[int, int, Path]] = []
        seen_roots: set[str] = set()

        for root in search_roots:
            try:
                resolved_root = root.resolve()
            except Exception:
                continue
            if not resolved_root.exists() or not resolved_root.is_dir():
                continue
            root_key = str(resolved_root)
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)

            try:
                dictionary_paths = list(resolved_root.rglob("*_feature_dictionary.csv"))
            except Exception:
                continue

            for dictionary_path in dictionary_paths:
                try:
                    dictionary_df = pd.read_csv(
                        dictionary_path,
                        dtype=str,
                        encoding="utf-8-sig",
                        usecols=["output_column"],
                    )
                except Exception:
                    continue

                output_columns = {
                    str(value).strip()
                    for value in dictionary_df["output_column"].dropna().tolist()
                    if str(value).strip()
                }
                if not normalized_feature_names.issubset(output_columns):
                    continue

                candidate_matches.append(
                    (
                        len(output_columns),
                        len(dictionary_path.parts),
                        dictionary_path,
                    )
                )

        if not candidate_matches:
            return None

        candidate_matches.sort(key=lambda item: (item[0], item[1], str(item[2])))
        return candidate_matches[0][2]

    def _build_feature_provenance_map(
        self,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        dictionary_path = self._find_feature_dictionary_path(dataset_name, dataset_path)
        if dictionary_path is None and feature_names:
            dictionary_path = self._find_feature_dictionary_path_by_feature_names(
                feature_names,
            )

        if dictionary_path is None:
            cache_key = f"missing::{dataset_name or ''}::{','.join(feature_names or [])}"
            cached_missing = self._feature_dictionary_cache.get(cache_key)
            if cached_missing is not None:
                return cached_missing
            self._feature_dictionary_cache[cache_key] = {}
            return {}

        return self._load_feature_provenance_from_path(dictionary_path)

    @staticmethod
    def _parse_value_label_options(raw_value_labels: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_parts = DatasetProvenanceMixin._split_dictionary_values(raw_value_labels)
        substantive_options: List[Dict[str, Any]] = []
        missing_options: List[Dict[str, Any]] = []

        for part in raw_parts:
            if "=" in part:
                raw_value, raw_label = part.split("=", 1)
                value = raw_value.strip()
                label = raw_label.strip()
            else:
                value = ""
                label = part.strip()

            option = {
                "value": value,
                "label": label,
            }
            if DatasetProvenanceMixin._is_missing_response_option(value, label):
                missing_options.append(option)
            else:
                substantive_options.append(option)

        return substantive_options, missing_options

    @staticmethod
    def _is_missing_response_option(value: str, label: str) -> bool:
        normalized_value = str(value).strip()
        normalized_label = str(label).strip().lower()

        try:
            numeric_value = float(normalized_value)
        except Exception:
            numeric_value = None

        if numeric_value is not None and numeric_value < 0:
            return True

        missing_markers = [
            "auslassen",
            "unklare beantwortung",
            "fragebogenrotation",
            "kein fragebogen",
            "fehlend",
            "missing",
            "nicht kalkulierbar",
            "ungueltig",
        ]
        return any(marker in normalized_label for marker in missing_markers)

    def _load_column_mapping_from_path(
        self,
        column_mapping_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        cache_key = str(column_mapping_path)
        cached = self._column_mapping_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            mapping_df = pd.read_csv(
                column_mapping_path,
                dtype=str,
                encoding="utf-8-sig",
            )
        except Exception:
            self._column_mapping_cache[cache_key] = {}
            return {}

        required_columns = {
            "original_spss_variable",
            "description",
            "value_labels",
        }
        if not required_columns.issubset(set(mapping_df.columns)):
            self._column_mapping_cache[cache_key] = {}
            return {}

        mapping: Dict[str, Dict[str, Any]] = {}
        for _, row in mapping_df.iterrows():
            variable_name = str(row.get("original_spss_variable", "") or "").strip()
            if not variable_name:
                continue

            response_options, missing_response_options = self._parse_value_label_options(
                row.get("value_labels"),
            )
            mapping[variable_name] = {
                "description": str(row.get("description", "") or "").strip(),
                "response_options": response_options,
                "missing_response_options": missing_response_options,
            }

        self._column_mapping_cache[cache_key] = mapping
        return mapping

    def _find_readable_column_mapping_path(
        self,
        dictionary_path: Path,
    ) -> Optional[Path]:
        direct_candidates = sorted(
            dictionary_path.parent.glob("*_readable_v1_column_mapping.csv")
        )
        if direct_candidates:
            return direct_candidates[0]

        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [repo_root, Path.home() / "Downloads"]
        seen: set[str] = set()

        for root in search_roots:
            try:
                resolved_root = root.resolve()
            except Exception:
                continue
            if not resolved_root.exists() or not resolved_root.is_dir():
                continue
            root_key = str(resolved_root)
            if root_key in seen:
                continue
            seen.add(root_key)
            try:
                matches = list(
                    resolved_root.rglob("*_readable_v1_column_mapping.csv")
                )
            except Exception:
                continue
            if matches:
                matches.sort(key=lambda path: len(path.parts))
                return matches[0]

        return None

    def _build_response_option_map_for_dictionary(
        self,
        dictionary_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        column_mapping_path = self._find_readable_column_mapping_path(dictionary_path)
        if column_mapping_path is None:
            return {}
        return self._load_column_mapping_from_path(column_mapping_path)

    def _find_feature_dictionary_path(
        self,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
    ) -> Optional[Path]:
        normalized_dataset_name = str(dataset_name or "").strip()
        if not normalized_dataset_name:
            return None

        expected_name = (
            f"{normalized_dataset_name[:-4]}_feature_dictionary.csv"
            if normalized_dataset_name.lower().endswith(".csv")
            else f"{normalized_dataset_name}_feature_dictionary.csv"
        )

        direct_candidates: List[Path] = []
        if dataset_path is not None:
            direct_candidates.append(dataset_path.parent / expected_name)
        direct_candidates.append(self.datasets_dir / expected_name)

        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate

        repo_root = Path(__file__).resolve().parents[2]
        search_roots = [repo_root, Path.home() / "Downloads"]
        seen: set[str] = set()

        for root in search_roots:
            try:
                resolved_root = root.resolve()
            except Exception:
                continue
            if not resolved_root.exists() or not resolved_root.is_dir():
                continue
            root_key = str(resolved_root)
            if root_key in seen:
                continue
            seen.add(root_key)
            try:
                matches = list(resolved_root.rglob(expected_name))
            except Exception:
                continue
            if matches:
                matches.sort(key=lambda path: len(path.parts))
                return matches[0]

        return None

    def _enrich_feature_schema_with_provenance(
        self,
        feature_schema: List[Dict[str, Any]],
        *,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
    ) -> List[Dict[str, Any]]:
        provenance_map = self._build_feature_provenance_map(
            dataset_name,
            dataset_path,
            [str(item.get("name", "") or "").strip() for item in feature_schema],
        )
        if not provenance_map:
            return feature_schema

        enriched_schema: List[Dict[str, Any]] = []
        for item in feature_schema:
            enriched_item = dict(item)
            feature_name = str(enriched_item.get("name", "") or "").strip()
            provenance = provenance_map.get(feature_name)
            if provenance:
                enriched_item["feature_provenance"] = provenance
                numeric_domain = self._infer_numeric_domain_from_provenance(
                    enriched_item,
                    provenance,
                )
                if numeric_domain is not None:
                    domain_min, domain_max = numeric_domain
                    enriched_item["min_value"] = domain_min
                    enriched_item["max_value"] = domain_max
                    default_value = enriched_item.get("default_value")
                    if default_value is not None:
                        try:
                            default_numeric = float(default_value)
                        except (TypeError, ValueError):
                            default_numeric = None
                        if default_numeric is not None:
                            enriched_item["default_value"] = round(
                                min(max(default_numeric, domain_min), domain_max),
                                2,
                            )
            enriched_schema.append(enriched_item)
        return enriched_schema

    @staticmethod
    def _infer_numeric_domain_from_provenance(
        feature_schema_entry: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        if str(feature_schema_entry.get("feature_type")) != "numeric":
            return None
        if str(provenance.get("construction_type")) != "item_mean":
            return None

        response_options = provenance.get("response_options") or []
        numeric_values: List[float] = []
        for option in response_options:
            if not isinstance(option, dict):
                continue
            try:
                numeric_value = float(option.get("value"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric_value):
                continue
            numeric_values.append(numeric_value)

        if len(numeric_values) < 2:
            return None

        return (float(min(numeric_values)), float(max(numeric_values)))
