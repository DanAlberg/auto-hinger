# app/agent_config.py

from dataclasses import dataclass
from typing import Dict, Any, Optional
import random

@dataclass 
class AgentConfig:
    """Configuration settings for the Hinge automation agent"""
    
    # Session settings
    max_profiles: int = 10
    max_retries_per_action: int = 3
    max_errors_before_abort: int = 5
    max_stuck_count: int = 3
    
    # Timing settings (in seconds)
    screenshot_delay: float = random.uniform(0, 1)
    action_delay: float = random.uniform(0, 1)
    navigation_delay: float = random.uniform(0, 1)
    text_input_delay: float = random.uniform(0, 1)
    keyboard_dismiss_delay: float = random.uniform(0, 1)
    
    # Profile analysis thresholds
    quality_threshold_high: int = 8
    quality_threshold_medium: int = 6
    conversation_threshold_high: int = 7
    min_positive_indicators: int = 2
    min_text_length_detailed: int = 200
    min_quality_for_detailed: int = 5
    
    # UI detection confidence thresholds
    min_button_confidence: float = 0.5
    min_ui_confidence: float = 0.7
    retry_confidence_threshold: float = 0.7
    
    # Swipe and tap coordinates (as percentages)
    dislike_button_coords: tuple = (0.15, 0.85)
    navigation_swipe_coords: tuple = (0.15, 0.5, 0.15, 0.375)  # x1, y1, x2, y2
    aggressive_swipe_coords: tuple = (0.9, 0.5, 0.1, 0.3)  # for unstuck
    scroll_area_coords: tuple = (0.5, 0.6)  # center x, y for scrolling
    keyboard_tap_area: tuple = (0.5, 0.25)  # tap outside keyboard
    
    # Fallback coordinates for comment interface
    fallback_comment_coords: tuple = (0.5, 0.75)
    fallback_send_coords: tuple = (0.75, 0.82)
    
    # Scroll settings
    max_scroll_attempts: int = 3
    scroll_distance_factor: float = 0.3  # how far to scroll
    vertical_swipe_x_pct: float = 0.20   # use ~20% of screen (move off extreme edge for better scroll classification)
    vertical_swipe_duration_ms: int = 600    # faster scroll (2x speed); still long enough to classify as scroll, not tap
    vertical_swipe_x_jitter_px: int = 3      # tiny horizontal jitter to avoid stationary-tap interpretation
    vertical_swipe_bottom_guard_pct: float = 0.90  # never place swipe endpoints below bottom 10% (avoid X button)
    
    # Horizontal content scraping
    max_horizontal_swipes: int = 8
    content_stable_repeats: int = 2
    horizontal_swipe_dx: tuple = (0.77, 0.23)  # start_x_percent, end_x_percent (10% shorter vs center)
    horizontal_swipe_duration_ms: int = 300    # faster horizontal swipe (2x speed)
    
    # CV+OCR biometrics extraction (horizontal row)
    use_cv_ocr_biometrics: bool = True
    cv_ocr_engine: str = "easyocr"  # "easyocr" | "tesseract"
    cv_band_height_ratio: float = 0.06
    cv_micro_swipe_ratio: float = 0.12
    cv_seek_swipe_ratio: float = 0.60
    cv_target_center_x_ratio: float = 0.30
    verbose_cv_timing: bool = True
    ignore_zodiac: bool = True  # only store zodiac_listed boolean

    # LLM payload selection
    llm_exclude_horizontal: bool = True  # exclude horizontal carousel frames from LLM payload
    
    # Age icon detection & carousel Y inference
    age_icon_roi: tuple = (0.1, 0.9)            # search age icon within this vertical ROI (top..bottom)
    age_icon_scales: tuple = (0.6, 1.5, 0.1)     # start, end, step for multi-scale matching
    age_icon_threshold: float = 0.42             # normalized threshold for match acceptance
    age_icon_use_edges: bool = True              # use Canny edges during matching
    age_icon_templates: tuple = ("assets/icon_age.png", "assets/icon_gender.png", "assets/icon_height.png")  # templates to use for age-row detection (2-of-3)
    age_dual_y_tolerance_px: int = 5             # absolute pixel tolerance for age/gender Y alignment
    age_dual_y_tolerance_ratio: float = 0.005    # relative tolerance (fraction of image height)
    require_both_icons_for_y: bool = True        # require both icons and average their Y when close
    icon_expected_px: int = 60               # expected on-screen icon height in pixels
    icon_scale_tolerance: float = 0.30       # +/- percentage around expected scale
    icon_min_px: int = 20                    # minimum scaled template height in pixels
    icon_max_roi_frac: float = 0.12          # max scaled template height as fraction of ROI height
    edges_dilate_iter: int = 1               # dilation iterations for edges match
    require_icon_detection_for_y: bool = True  # if True, do not fall back to edge-only Y inference
    icon_high_threshold: float = 0.80          # high-confidence threshold for sweep detection
    max_icon_sweep_pages: int = 8              # max pages to scroll during icon sweep before giving up

    carousel_y_roi_top: float = 0.10             # top of ROI for edge-based Y inference
    carousel_y_roi_bottom: float = 0.90          # bottom of ROI for edge-based Y inference
    carousel_y_smooth_kernel: int = 21           # smoothing kernel for row-strength profiling

    # Carousel vertical seek scanning options
    max_carousel_y_scans: int = 2                               # max small peek scrolls when age row not visible
    carousel_scan_step: tuple = (0.72, 0.48)                    # peek swipe: start_y_pct -> end_y_pct
    carousel_restore_after_seek: bool = True                    # restore viewport after successful seek
    carousel_detection_roi_stages: tuple = (                    # progressive ROIs for detection
        (0.1, 0.9),
    )
    carousel_detection_threshold_decay: float = 0.05            # lower template threshold per stage

    # Image dedup/stabilization (aHash) & paging
    image_hash_size: int = 8                     # aHash size (NxN); 8 -> 64-bit
    image_hash_threshold: int = 5                # Hamming distance threshold for "similar"
    image_stable_repeats: int = 2                # stop after N consecutive duplicates

    # Horizontal carousel-specific stabilization (overrides when present)
    hscroll_hash_threshold: int = 1              # stricter Hamming distance for horizontal carousel dedup
    hscroll_stable_repeats: int = 1              # stop after 1 duplicate frame on horizontal carousel
    hscroll_hash_roi_ratio: float = 0.04         # narrow vertical band (~4%) for horizontal aHash ROI (markers strip)

    max_vertical_pages: int = 10                 # vertical pages cap before giving up
    
    # Recovery strategies
    enable_aggressive_navigation: bool = True
    enable_back_button_recovery: bool = True
    enable_keyboard_recovery: bool = True
    
    # Comment generation
    default_comment: str = "Hey, I'd love to meet up!"
    comment_style: str = "balanced"  # comedic, flirty, straightforward, balanced
    confirm_before_send: bool = True  # prompt before tapping Send Like
    
    # Debug settings
    save_screenshots: bool = True
    screenshot_dir: str = "images"
    verbose_logging: bool = True

    # Manual confirmation mode
    manual_confirm: bool = False
    manual_log_dir: str = "logs"

    # AI tracing
    ai_trace: bool = True
    ai_trace_log_dir: str = "logs"

    # Extraction models (LLM)
    extraction_model: str = "gpt-5"
    extraction_small_model: str = "gpt-5-mini"  # for small/logical tasks
    extraction_retry: int = 1
    llm_max_images: int = 10  # hard cap for LLM image submissions

    # Testing and typing verification
    dry_run: bool = False  # when True, never send likes/comments (skips Send taps)
    verify_typed_text: bool = False  # optional OCR sanity check after typing (off by default)
    
    # Deterministic control & settings
    deterministic_mode: bool = True
    like_mode: str = "priority"  # "priority" or "normal"
    scrape_only: bool = False
    export_dir: str = "logs"
    precheck_strict: bool = True
    sync_check_at_stages: bool = True
    max_priority_likes_per_session: int = 999999
    
    # Device settings
    device_ip: str = "127.0.0.1"
    adb_port: int = 5037
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


# Default configuration
DEFAULT_CONFIG = AgentConfig()

# High-performance configuration (faster, less thorough)
FAST_CONFIG = AgentConfig(
    max_profiles=20,
    max_retries_per_action=2,
    screenshot_delay=0.5,
    action_delay=1.0,
    navigation_delay=2.0,
    max_scroll_attempts=2,
    quality_threshold_high=7,
    quality_threshold_medium=5,
    conversation_threshold_high=6
)
