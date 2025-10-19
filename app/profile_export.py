import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Try xlwings first for live updates to an open workbook (Excel COM on Windows)
try:
    import xlwings as xw  # type: ignore
except Exception:
    xw = None  # Fallback to openpyxl if xlwings is unavailable

from openpyxl import Workbook, load_workbook


# Canonical, scan-derived-only schema (no scores, completeness, etc.)
SCHEMA_V2: List[str] = [
    # Identity
    "name",
    "age",
    "height",
    "location",
    # Profile
    "sexuality",
    "ethnicity",
    "current_children",
    "family_plans",
    "covid_vaccine",
    "zodiac_sign",
    "hometown",
    # Education / work / beliefs
    "university",
    "job_title",
    "work",
    "religious_beliefs",
    # Politics / languages / relationship
    "politics",
    "languages_spoken",
    "dating_intentions",
    "relationship_type",
    # Lifestyle (tri-state)
    "drinking",
    "smoking",
    "marijuana",
    "drugs",
    # Pets (boolean-ish)
    "pets_dog",
    "pets_cat",
    "pets_bird",
    "pets_fish",
    "pets_reptile",
    # Content
    "bio",
    "prompts_and_answers",
    "interests",
    "summary",
]


class ProfileExporter:
    """
    Excel exporter with incremental writes that can update while the workbook is open.
    Prefers xlwings (live write via Excel COM); falls back to openpyxl if xlwings is unavailable.
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

        # Use canonical scan-derived schema by default
        self.schema = schema or SCHEMA_V2

        # Workbook path at repository root (hinge-automation/profiles.xlsx)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.persistent_xlsx_path = os.path.join(repo_root, "profiles.xlsx")
        self.sheet_name = "profiles"  # user will clear old sheet; keep name stable

        # Initialize workbook and header; seed last-row signature
        self._last_xlsx_signature: Optional[Tuple[str, str, str]] = None
        if self.export_xlsx:
            self._ensure_xlsx_header()
            self._seed_last_signature_from_xlsx()

    # ===== Schema/header management =====

    def _ensure_xlsx_header(self) -> None:
        """Ensure profiles.xlsx and header row exist; upgrade header to current schema if mismatched."""
        try:
            if xw is not None:
                opened_here = False
                wb = self._get_or_open_book_via_xlwings()
                if wb is None:
                    # If xlwings path fails entirely, drop to openpyxl
                    self._ensure_header_openpyxl()
                    return
                ws = self._get_or_create_sheet_xlwings(wb, self.sheet_name)
                # Read current header (row 1)
                current_header = self._read_header_xlwings(ws)
                needs_upgrade = (
                    current_header is None
                    or len(current_header) != len(self.schema)
                    or any((current_header[i] or "") != self.schema[i] for i in range(len(self.schema)))
                )
                if needs_upgrade:
                    # Write header horizontally
                    ws.range(1, 1).value = [self.schema]
                # Save when we opened the workbook; if attached to an existing Excel instance, saving is safe.
                wb.save()
                # If we created a hidden Excel app, close it to avoid orphaned processes.
                self._maybe_close_book_if_opened_here(wb)
                return

            # Fallback: openpyxl path
            self._ensure_header_openpyxl()
        except Exception as e:
            print(f"Warning: Failed to initialize XLSX header: {e}")

    def _ensure_header_openpyxl(self) -> None:
        """Ensure header using openpyxl (fallback path)."""
        try:
            if not os.path.exists(self.persistent_xlsx_path):
                wb = Workbook()
                ws = wb.active
                ws.title = self.sheet_name
                ws.append(self.schema)
                wb.save(self.persistent_xlsx_path)
                wb.close()
            else:
                wb = load_workbook(self.persistent_xlsx_path)
                ws = wb.active
                # Ensure we are on the correct sheet name
                if ws.title != self.sheet_name:
                    # Try to get or create
                    ws = wb[self.sheet_name] if self.sheet_name in wb.sheetnames else wb.create_sheet(self.sheet_name)
                # If empty, write header
                if ws.max_row == 0:
                    ws.append(self.schema)
                    wb.save(self.persistent_xlsx_path)
                else:
                    # Compare/upgrade header row
                    header_cells = next(ws.iter_rows(min_row=1, max_row=1, values_only=False))
                    current_header = [cell.value if cell.value is not None else "" for cell in header_cells]
                    needs_upgrade = (
                        len(current_header) != len(self.schema)
                        or any((current_header[i] or "") != self.schema[i] for i in range(len(self.schema)))
                    )
                    if needs_upgrade:
                        for i, name in enumerate(self.schema, start=1):
                            ws.cell(row=1, column=i).value = name
                        wb.save(self.persistent_xlsx_path)
                wb.close()
        except Exception as e:
            print(f"Warning: Failed to initialize XLSX (openpyxl): {e}")

    def _read_header_xlwings(self, ws) -> Optional[List[str]]:
        try:
            # Expand across the first row to the right
            rng = ws.range(1, 1).expand("right")
            vals = rng.value
            if vals is None:
                return None
            # xlwings returns a scalar for single cell, list for rows
            if isinstance(vals, list):
                return [str(v) if v is not None else "" for v in vals]
            return [str(vals)]
        except Exception:
            return None

    # ===== Signature management (consecutive duplicate guard) =====

    def _seed_last_signature_from_xlsx(self) -> None:
        """Read the last data row to initialize the duplicate guard signature."""
        try:
            if not os.path.exists(self.persistent_xlsx_path):
                self._last_xlsx_signature = None
                return
            wb = load_workbook(self.persistent_xlsx_path, read_only=True)
            ws = wb[self.sheet_name] if self.sheet_name in wb.sheetnames else wb.active
            # Read header names from first row
            header_vals = [cell.value if cell.value is not None else "" for cell in next(ws.iter_rows(min_row=1, max_row=1, values_only=False))]
            def idx(col: str) -> Optional[int]:
                try:
                    return header_vals.index(col)
                except ValueError:
                    return None
            idx_name = idx("name")
            idx_age = idx("age")
            idx_height = idx("height")

            last = None
            if ws.max_row and ws.max_row >= 2:
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
        age = row.get("age", "")
        height = row.get("height", "")
        return (str(name).strip().lower(), str(age).strip(), str(height).strip())

    # ===== Append logic =====

    def append_row(self, row: Dict[str, Any]) -> None:
        """
        Append a single profile row to Excel with a consecutive-duplicate guard.
        Skip only if the last written row has the same (name, age, height).
        """
        if not self.export_xlsx:
            return

        try:
            sig = self._compute_signature(row)

            # Prefer xlwings for live updates to an open workbook
            if xw is not None:
                wb = self._get_or_open_book_via_xlwings()
                if wb is not None:
                    ws = self._get_or_create_sheet_xlwings(wb, self.sheet_name)
                    # Ensure header
                    current_header = self._read_header_xlwings(ws) or []
                    if current_header != self.schema:
                        ws.range(1, 1).value = [self.schema]

                    # Check last row to guard consecutive duplicates
                    last_row_num = self._xlwings_last_data_row(ws)
                    if last_row_num >= 2:
                        last_vals = ws.range(last_row_num, 1).expand("right").value
                        if not isinstance(last_vals, list):
                            last_vals = [last_vals]
                        # Map to header
                        def idx(col: str) -> Optional[int]:
                            try:
                                return self.schema.index(col)
                            except ValueError:
                                return None
                        idx_name = idx("name")
                        idx_age = idx("age")
                        idx_height = idx("height")
                        last_name = (last_vals[idx_name] if idx_name is not None and idx_name < len(last_vals) else "") if last_vals else ""
                        last_age = (last_vals[idx_age] if idx_age is not None and idx_age < len(last_vals) else "") if last_vals else ""
                        last_height = (last_vals[idx_height] if idx_height is not None and idx_height < len(last_vals) else "") if last_vals else ""
                        last_sig = (
                            str(last_name or "").strip().lower(),
                            str(last_age or "").strip(),
                            str(last_height or "").strip(),
                        )
                        if last_sig == sig:
                            # No-op
                            wb.save()
                            self._maybe_close_book_if_opened_here(wb)
                            return

                    next_row = 2 if last_row_num < 1 else (last_row_num + 1)
                    values = [row.get(col, "") for col in self.schema]
                    # Write horizontally in a single operation
                    ws.range(next_row, 1).value = [values]
                    wb.save()
                    self._last_xlsx_signature = sig
                    self._maybe_close_book_if_opened_here(wb)
                    return

            # Fallback to openpyxl
            self._append_row_openpyxl(row)
            self._last_xlsx_signature = sig
        except Exception as e:
            # Non-fatal: log and continue
            print(f"Warning: Failed to append to XLSX: {e}")

    def _append_row_openpyxl(self, row: Dict[str, Any]) -> None:
        try:
            if not os.path.exists(self.persistent_xlsx_path):
                wb = Workbook()
                ws = wb.active
                ws.title = self.sheet_name
                ws.append(self.schema)
            else:
                wb = load_workbook(self.persistent_xlsx_path)
                ws = wb[self.sheet_name] if self.sheet_name in wb.sheetnames else wb.active
                if ws.max_row == 0:
                    ws.append(self.schema)
            ws.append([row.get(col, "") for col in self.schema])
            wb.save(self.persistent_xlsx_path)
            wb.close()
        except Exception as e:
            print(f"Warning: openpyxl append failed: {e}")

    # ===== xlwings helpers =====

    def _get_or_open_book_via_xlwings(self):
        """Attach to an already open workbook if available; otherwise open it. Returns a Book or None."""
        try:
            # Try to find an existing open workbook by fullname
            full = os.path.abspath(self.persistent_xlsx_path)
            for app in xw.apps:
                try:
                    for b in app.books:
                        try:
                            if os.path.abspath(b.fullname).lower() == full.lower():
                                # Attach to existing - mark as not opened here
                                setattr(b, "_opened_here", False)
                                return b
                        except Exception:
                            continue
                except Exception:
                    continue
            # If not found, open (will start Excel if needed)
            wb = xw.Book(full)  # opens or attaches
            setattr(wb, "_opened_here", True)
            return wb
        except Exception:
            return None

    def _get_or_create_sheet_xlwings(self, wb, sheet_name: str):
        try:
            return wb.sheets[sheet_name]
        except Exception:
            try:
                return wb.sheets.add(sheet_name, after=wb.sheets[-1])
            except Exception:
                # Fallback to the first sheet
                return wb.sheets[0]

    def _xlwings_last_data_row(self, ws) -> int:
        """Get the last non-empty row index in column A (1-based)."""
        try:
            # Excel max rows for .xlsx is 1,048,576
            last = ws.range("A1048576").end("up").row
            return int(last or 1)
        except Exception:
            return 1

    def _maybe_close_book_if_opened_here(self, wb) -> None:
        """Close the workbook if our process opened it (avoid leaving hidden Excel instances)."""
        try:
            opened_here = getattr(wb, "_opened_here", False)
            if opened_here:
                # Save already called by caller; just close the book to release the COM instance
                wb.close()
        except Exception:
            pass

    # ===== Public API =====

    def close(self) -> None:
        # No persistent handles; nothing to close explicitly.
        pass

    def get_paths(self) -> Dict[str, str]:
        return {
            "xlsx": self.persistent_xlsx_path if self.export_xlsx else "",
        }
