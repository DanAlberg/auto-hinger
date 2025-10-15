import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import pandas as pd  # Optional; used only if export_xlsx=True and pandas is available
except Exception:
    pd = None


DEFAULT_SCHEMA: List[str] = [
    "session_id",
    "timestamp",
    "profile_index",
    # Extracted identity
    "name",
    "estimated_age",
    "location",
    "profession",
    "education",
    # Lifestyle / attributes
    "drinks",
    "smokes",
    "cannabis",
    "drugs",
    "religion",
    "politics",
    "kids",
    "wants_kids",
    "height",
    "languages",
    "interests",
    "attribute_chips_raw",
    # Content metrics
    "prompts_count",
    "extracted_text_length",
    "content_depth",
    "completeness",
    # Analyzer outputs
    "profile_quality_score",
    "conversation_potential",
    "should_like",
    "policy_reason",
    # Action details
    "like_mode",
    "sent_like",
    "sent_comment",
    "comment_id",
    "comment_hash",
    # Trace
    "screenshot_path",
    "errors_encountered",
]


class ProfileExporter:
    """
    Streams profile rows to CSV and optionally writes an XLSX at session end.
    - CSV is written incrementally for robustness.
    - XLSX is written at close() if enabled and pandas is available.
    """

    def __init__(
        self,
        export_dir: str,
        session_id: str,
        export_csv: bool = True,
        export_xlsx: bool = False,
        schema: Optional[List[str]] = None,
    ) -> None:
        self.export_dir = export_dir
        self.session_id = session_id
        self.export_csv = export_csv
        self.export_xlsx = export_xlsx and (pd is not None)

        self.schema = schema or DEFAULT_SCHEMA
        self.rows_buffer: List[Dict[str, Any]] = []

        os.makedirs(self.export_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.export_dir, f"profiles_{self.session_id}.csv")
        self.xlsx_path = os.path.join(self.export_dir, f"profiles_{self.session_id}.xlsx")
        # Persistent workbook at repository root (auto-hinger/profiles.xlsx)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.persistent_xlsx_path = os.path.join(repo_root, "profiles.xlsx")

        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False

        if self.export_csv:
            # Open CSV in append mode; write header on first append
            self._csv_file = open(self.csv_path, mode="a", encoding="utf-8-sig", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.schema, extrasaction="ignore")

    def _ensure_csv_header(self) -> None:
        if self.export_csv and self._csv_writer and not self._csv_header_written:
            self._csv_writer.writeheader()
            self._csv_header_written = True

    def append_row(self, row: Dict[str, Any]) -> None:
        """
        Append a single profile row.
        Missing keys will be filled with empty strings for CSV schema compatibility.
        """
        # Normalize row to schema for CSV
        csv_row = {k: row.get(k, "") for k in self.schema}

        if self.export_csv and self._csv_writer:
            self._ensure_csv_header()
            self._csv_writer.writerow(csv_row)
            # Flush quickly to avoid data loss if interrupted
            try:
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
            except Exception:
                pass

        # Always buffer rows for potential XLSX output
        self.rows_buffer.append(row.copy())

    def close(self) -> None:
        # Close CSV handle
        try:
            if self._csv_file:
                self._csv_file.close()
        except Exception:
            pass
        finally:
            self._csv_file = None
            self._csv_writer = None

        # Optionally write XLSX at the end using pandas
        if self.export_xlsx and pd is not None:
            try:
                # Build a superset of keys across rows to preserve all columns
                all_keys = set(self.schema)
                for r in self.rows_buffer:
                    all_keys.update(r.keys())
                ordered_cols = [c for c in self.schema] + [k for k in all_keys if k not in self.schema]

                new_df = pd.DataFrame(self.rows_buffer)
                # Reorder columns for new data buffer first
                new_df = new_df.reindex(columns=ordered_cols)
                # Append to a persistent workbook (auto-hinger/profiles.xlsx)
                if os.path.exists(self.persistent_xlsx_path):
                    try:
                        existing_df = pd.read_excel(self.persistent_xlsx_path)
                        # Union columns between existing and new
                        all_cols = list(dict.fromkeys(list(existing_df.columns) + list(new_df.columns)))
                        existing_df = existing_df.reindex(columns=all_cols)
                        new_df = new_df.reindex(columns=all_cols)
                        out_df = pd.concat([existing_df, new_df], ignore_index=True)
                    except Exception:
                        # If existing file unreadable, fall back to new only
                        out_df = new_df
                else:
                    out_df = new_df
                # Write back to persistent path
                out_df.to_excel(self.persistent_xlsx_path, index=False)
            except Exception as e:
                # XLSX is best-effort; do not raise
                print(f"Warning: Failed to write XLSX export: {e}")

    def get_paths(self) -> Dict[str, str]:
        return {
            "csv": self.csv_path if self.export_csv else "",
            "xlsx": self.persistent_xlsx_path if (self.export_xlsx and pd is not None) else "",
        }
