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
    ai_trace: bool = False
    ai_trace_log_dir: str = "logs"
    # Testing and typing verification
    dry_run: bool = False  # when True, never send likes/comments (skips Send taps)
    verify_typed_text: bool = False  # optional OCR sanity check after typing (off by default)
    
    # Deterministic control & export settings
    deterministic_mode: bool = True
    like_mode: str = "priority"  # "priority" or "normal"
    export_xlsx: bool = True
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
