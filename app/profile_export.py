import os
from typing import Dict, Any, Optional
from datetime import datetime
import json

from sqlite_store import init_db, upsert_profile_flat, get_db_path, get_profile_id_by_unique
from profile_eval import evaluate_profile_fields, compute_profile_score


class ProfileExporter:
    """
    SQLite exporter (flat column schema).
    - Writes one row per profile into profiles.db at the repo root.
    - Input:
      • Structured payload: {"extracted_profile": {...}, "analysis": {...?}, "metadata": {...?}}
      • Legacy: a plain extracted profile dict (then analysis is computed here)
    - Dedup: UNIQUE(Name COLLATE NOCASE, Age, Height_cm) via UPSERT DO NOTHING.
    """

    def __init__(self, export_dir: str, session_id: str) -> None:
        # session_id kept only for compatibility with construction sites; not stored
        self.export_dir = export_dir
        self.session_id = session_id
        self.db_path = get_db_path()
        init_db(self.db_path)

    def append_row(self, data: Dict[str, Any]) -> Optional[int]:
        if not data:
            return

        # Unpack payload
        if isinstance(data, dict) and (
            "extracted_profile" in data or "analysis" in data or "metadata" in data
        ):
            extracted: Dict[str, Any] = data.get("extracted_profile") or {}
            analysis: Optional[Dict[str, Any]] = data.get("analysis")
            metadata: Dict[str, Any] = data.get("metadata") or {}
        else:
            extracted = data if isinstance(data, dict) else {}
            analysis = None
            metadata = {}

        # Timestamp: prefer provided; else now
        ts = metadata.get("timestamp") or datetime.now().isoformat(timespec="seconds")

        # Ensure analysis exists
        if not analysis:
            try:
                analysis = evaluate_profile_fields(extracted)
            except Exception:
                analysis = {}

        # Compute score and breakdown
        try:
            scoring = compute_profile_score(extracted, analysis or {})
            score = int(scoring.get("score", 0))
            score_breakdown = json.dumps(scoring.get("contributors") or scoring.get("top_contributors") or [], ensure_ascii=False)
        except Exception:
            score = 0
            score_breakdown = "[]"

        # UPSERT flattened row
        try:
            rowid = upsert_profile_flat(
                extracted_profile=extracted,
                enrichment=analysis or {},
                score=score,
                score_breakdown=score_breakdown,
                timestamp=ts,
                db_path=self.db_path,
            )
            pid: Optional[int] = None
            if rowid is not None:
                pid = int(rowid)
                print(f"✅ Saved profile to {self.db_path} (rowid={rowid}, score={score})")
            else:
                # Duplicate: resolve existing id via UNIQUE key (Name, Age, Height_cm)
                try:
                    name = (extracted.get("Name") or "").strip()
                    age_v = extracted.get("Age")
                    height_v = extracted.get("Height") if extracted.get("Height") is not None else extracted.get("Height_cm")
                    age_i = int(float(age_v)) if age_v is not None and str(age_v).strip() != "" else None
                    height_i = int(float(height_v)) if height_v is not None and str(height_v).strip() != "" else None
                    if name and age_i is not None and height_i is not None:
                        pid = get_profile_id_by_unique(name, age_i, height_i, db_path=self.db_path)
                except Exception:
                    pid = None
                print("🟨 Duplicate ignored (same Name/Age/Height_cm)")
            return pid
        except ValueError as ve:
            # Height/Age missing or invalid, or Name empty
            print(f"❌ Export validation error: {ve}")
            return None
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None

    def get_paths(self) -> Dict[str, str]:
        return {"db": self.db_path}

    def close(self) -> None:
        pass
