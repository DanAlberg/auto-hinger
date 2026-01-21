# app/analyzer.py
# Facade that exposes analyzer functions with existing names.
# Currently routes to LLM-backed implementations in analyzer_openai.py.
# This allows incremental migration without changing all call sites.

from analyzer_openai import (
    extract_text_from_image,
    generate_comment,
    generate_contextual_date_comment,
    analyze_dating_ui,
    find_ui_elements,
    analyze_profile_scroll_content,
    get_profile_navigation_strategy,
    detect_comment_ui_elements,
    verify_action_success,
)

__all__ = [
    "extract_text_from_image",
    "generate_comment",
    "generate_contextual_date_comment",
    "analyze_dating_ui",
    "find_ui_elements",
    "analyze_profile_scroll_content",
    "get_profile_navigation_strategy",
    "detect_comment_ui_elements",
    "verify_action_success",
]
