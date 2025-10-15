import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from openpyxl import Workbook, load_workbook


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
    Excel-only exporter with incremental writes:
    - Writes a row to auto-hinger/profiles.xlsx after each profile.
    - Guards against consecutive duplicate rows using (name, estimated_age, height).
    """

    def __init__(
        self,
        export_dir: str,
        session_id: str,
        export_xlsx: bool = True,
        schema: Optional[List[str]] = None,
    ) -> None:
        self.export_dir = export_dir
        self.session_id = session_id
        self.export_xlsx = export_xlsx

        self.schema = schema or DEFAULT_SCHEMA

        # Persistent workbook at repository root (auto-hinger/profiles.xlsx)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.persistent_xlsx_path = os.path.join(repo_root, "profiles.xlsx")

        # Initialize workbook and header, and seed last-row signature
        self._last_xlsx_signature: Optional[Tuple[str, str, str]] = None
        if self.export_xlsx:
            self._ensure_xlsx_header()
            self._seed_last_signature_from_xlsx()

    def _ensure_xlsx_header(self) -> None:
        """Ensure profiles.xlsx exists and has header row."""
        try:
            if not os.path.exists(self.persistent_xlsx_path):
                wb = Workbook()
                ws = wb.active
                ws.title = "profiles"
                ws.append(self.schema)
                wb.save(self.persistent_xlsx_path)
                wb.close()
            else:
                wb = load_workbook(self.persistent_xlsx_path)
                ws = wb.active
                # If workbook exists but has no rows, add header
                if ws.max_row == 0:
                    ws.append(self.schema)
                    wb.save(self.persistent_xlsx_path)
                wb.close()
        except Exception as e:
            print(f"Warning: Failed to initialize XLSX: {e}")

    def _seed_last_signature_from_xlsx(self) -> None:
        """Read the last data row from XLSX to initialize the duplicate guard signature."""
        try:
            if not os.path.exists(self.persistent_xlsx_path):
                self._last_xlsx_signature = None
                return
            wb = load_workbook(self.persistent_xlsx_path, read_only=True)
            ws = wb.active
            if ws.max_row and ws.max_row >= 2:
                # Read header names from first row
                header_vals = [cell.value if cell.value is not None else "" for cell in next(ws.iter_rows(min_row=1, max_row=1, values_only=False))]
                def idx(col: str) -> Optional[int]:
                    try:
                        return header_vals.index(col)
                    except ValueError:
                        return None
                idx_name = idx("name")
                idx_age = idx("estimated_age")
                idx_height = idx("height")
                # Iterate to last row (read_only mode)
                last = None
                for row in ws.iter_rows(min_row=2, values_only=True):
                    last = row
                if last is not None:
                    name_val = (last[idx_name] if idx_name is not None and idx_name < len(last) else "") if last else ""
                    age_val = (last[idx_age] if idx_age is not None and idx_age < len(last) else "") if last else ""
                    height_val = (last[idx_height] if idx_height is not None and idx_height < len(last) else "") if last else ""
                    self._last_xlsx_signature = (
                        str(name_val or "").strip().lower(),
                        str(age_val or "").strip(),
                        str(height_val or "").strip(),
                    )
            wb.close()
        except Exception:
            # Non-fatal: if unreadable, leave as None
            self._last_xlsx_signature = None

    def _compute_signature(self, row: Dict[str, Any]) -> Tuple[str, str, str]:
        name = (row.get("name", "") or "")
        age = row.get("estimated_age", "")
        height = row.get("height", "")
        return (str(name).strip().lower(), str(age).strip(), str(height).strip())

    def append_row(self, row: Dict[str, Any]) -> None:
        """
        Append a single profile row to Excel with a consecutive-duplicate guard.
        - Skip only if the last written row has the same (name, age, height) AND the same sent_like/sent_comment flags.
        - If action flags changed (e.g., sent_like flips from 0 to 1), allow appending.
        """
        if not self.export_xlsx:
            return

        try:
            sig = self._compute_signature(row)
            wb = load_workbook(self.persistent_xlsx_path)
            ws = wb.active

            # Build header index map
            header_cells = next(ws.iter_rows(min_row=1, max_row=1, values_only=False))
            header_vals = [cell.value if cell.value is not None else "" for cell in header_cells]
            def idx(col: str) -> Optional[int]:
                try:
                    return header_vals.index(col)
                except ValueError:
                    return None
            idx_name = idx("name")
            idx_age = idx("estimated_age")
            idx_height = idx("height")
            idx_sent_like = idx("sent_like")
            idx_sent_comment = idx("sent_comment")

            # Check last row for consecutive-duplicate (including action flags)
            should_skip = False
            if ws.max_row and ws.max_row >= 2:
                last = None
                for r in ws.iter_rows(min_row=2, values_only=True):
                    last = r
                if last is not None:
                    last_name = (last[idx_name] if idx_name is not None and idx_name < len(last) else "") if last else ""
                    last_age = (last[idx_age] if idx_age is not None and idx_age < len(last) else "") if last else ""
                    last_height = (last[idx_height] if idx_height is not None and idx_height < len(last) else "") if last else ""
                    last_sig = (
                        str(last_name or "").strip().lower(),
                        str(last_age or "").strip(),
                        str(last_height or "").strip(),
                    )
                    if last_sig == sig:
                        # Compare action flags; treat missing as 0
                        last_like = last[idx_sent_like] if (idx_sent_like is not None and idx_sent_like < len(last)) else 0
                        last_comment = last[idx_sent_comment] if (idx_sent_comment is not None and idx_sent_comment < len(last)) else 0
                        new_like = row.get("sent_like", 0) or 0
                        new_comment = row.get("sent_comment", 0) or 0
                        if int(last_like or 0) == int(new_like or 0) and int(last_comment or 0) == int(new_comment or 0):
                            should_skip = True

            if should_skip:
                wb.close()
                return

            # Append row
            ws.append([row.get(col, "") for col in self.schema])
            wb.save(self.persistent_xlsx_path)
            wb.close()
            self._last_xlsx_signature = sig
        except Exception as e:
            # Non-fatal: log and continue
            print(f"Warning: Failed to append to XLSX: {e}")

    def close(self) -> None:
        # No persistent handles; nothing to close for XLSX.
        pass

    def get_paths(self) -> Dict[str, str]:
        return {
            "xlsx": self.persistent_xlsx_path if self.export_xlsx else "",
        }
