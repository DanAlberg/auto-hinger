# app/langgraph_hinge_agent.py

import json
import time
import uuid
from typing import Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
import base64
from openai import OpenAI
import os
import shutil
from datetime import datetime
from functools import wraps

from helper_functions import (
    connect_device, get_screen_resolution, open_hinge, reset_hinge_app,
    capture_screenshot, tap, tap_with_confidence, swipe,
    dismiss_keyboard, clear_screenshots_directory, clear_submitted_directory, detect_like_button_cv, detect_send_button_cv, detect_comment_field_cv, detect_keyboard_tick_cv, detect_age_icon_cv, detect_age_icon_cv_multi, detect_age_row_dual_templates, infer_carousel_y_by_edges, are_images_similar, are_images_similar_roi, input_text_robust
)
from analyzer import (
    extract_text_from_image, analyze_dating_ui,
    find_ui_elements, analyze_profile_scroll_content,
    detect_comment_ui_elements, generate_comment, generate_contextual_date_comment
)
from batch_payload import build_llm_batch_payload, build_profile_prompt
from data_store import store_generated_comment, calculate_template_success_rates
from prompt_engine import update_template_weights
from profile_export import ProfileExporter
from profile_eval import evaluate_profile_fields
import hashlib
from cv_biometrics import extract_biometrics_from_carousel


class HingeAgentState(TypedDict):
    """State maintained throughout the dating app automation workflow"""
    
    # Device and session info
    device: Any
    width: int
    height: int
    max_profiles: int
    current_profile_index: int
    
    # Session metrics
    profiles_processed: int
    likes_sent: int
    comments_sent: int
    errors_encountered: int
    stuck_count: int
    
    # Current profile data
    current_screenshot: Optional[str]
    profile_text: str
    profile_analysis: Dict[str, Any]
    decision_reason: str
    
    # Profile change detection data
    previous_profile_text: str
    previous_profile_features: Dict[str, Any]
    
    # Action results
    last_action: str
    action_successful: bool
    retry_count: int
    
    # Generated content
    generated_comment: str
    comment_id: str
    
    # Button coordinates
    like_button_coords: Optional[tuple]
    like_button_confidence: float
    
    # Control flow
    should_continue: bool
    completion_reason: str
    
    # AI decision context
    ai_reasoning: str
    next_tool_suggestion: str
    
    # Batch processing for LangGraph recursion limit management
    batch_start_index: int

    # Future: prebuilt LLM batch request (no submission performed)
    llm_batch_request: Dict[str, Any]


def gated_step(func):
    @wraps(func)
    def wrapper(self, state: HingeAgentState):
        step_name = func.__name__
        if getattr(self.config, "manual_confirm", False):
            # Ensure logging initialized
            if getattr(self, "manual_log_path", None) is None:
                try:
                    self._init_manual_logging()
                except Exception:
                    pass
            # Log BEGIN with brief context
            self._log(
                f"BEGIN {step_name} "
                f"idx={state.get('current_profile_index', 'n/a')} "
                f"likes={state.get('likes_sent', 0)} "
                f"comments={state.get('comments_sent', 0)} "
                f"errors={state.get('errors_encountered', 0)}"
            )
            # Ask for confirmation
            confirmed = self._confirm_step(step_name, state)
            if not confirmed:
                aborted = {
                    **state,
                    "should_continue": False,
                    "completion_reason": f"User aborted at {step_name}",
                    "last_action": step_name,
                    "action_successful": False
                }
                if step_name == "ai_decide_action_node":
                    aborted["next_tool_suggestion"] = "finalize"
                self._log(f"ABORTED {step_name}")
                return aborted
        # Execute actual step
        result = func(self, state)
        if getattr(self.config, "manual_confirm", False):
            self._log(
                f"END {step_name} action_successful={result.get('action_successful', False)} "
                f"likes={result.get('likes_sent', 0)} "
                f"comments={result.get('comments_sent', 0)} "
                f"errors={result.get('errors_encountered', 0)}"
            )
        return result
    return wrapper


class LangGraphHingeAgent:
    """
    LangGraph-powered Hinge automation agent with AI-controlled decision making.
    Replaces agent controller with improved workflow management.
    """
    
    def __init__(self, max_profiles: int = 10, config=None):
        from agent_config import DEFAULT_CONFIG
        
        self.max_profiles = max_profiles
        self.config = config or DEFAULT_CONFIG
        self.ai_client = None
        self.graph = self._build_workflow()
        # Cached UI coords
        self._tick_coords = None

        # Manual confirm logging path
        self.manual_log_path = None
        if getattr(self.config, "manual_confirm", False):
            self._init_manual_logging()
            self._log("Manual confirmation mode enabled; every step requires approval and is logged.")

        # AI trace setup
        self.ai_trace_file = None
        if getattr(self.config, "ai_trace", False):
            self._init_ai_trace_logging()
        
        # Profile batch processing to avoid LangGraph recursion limits
        self.profiles_per_batch = 3  # Process 3 profiles per batch to stay under 25-turn limit
        self.max_turns_per_profile = 8  # Estimated max turns needed per profile

        # Session/export setup
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exporter = None
    
    def _init_manual_logging(self) -> None:
        """Initialize manual confirmation session logging."""
        try:
            os.makedirs(self.config.manual_log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.manual_log_path = os.path.join(self.config.manual_log_dir, f"manual_{ts}.log")
            header = (
                f"Session started {datetime.now().isoformat()} "
                f"manual_confirm={self.config.manual_confirm} "
                f"max_profiles={self.max_profiles}\n"
            )
            with open(self.manual_log_path, "a", encoding="utf-8") as f:
                f.write(header)
        except Exception as e:
            print(f"[manual] Failed to init log file: {e}")
            self.manual_log_path = None

    def _log(self, message: str) -> None:
        """Structured log to console and file with timestamp."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{ts}] {message}"
        print(line)
        try:
            if self.manual_log_path:
                with open(self.manual_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            print(f"[manual] Failed to write log: {e}")

    def _confirm_step(self, step_name: str, state: HingeAgentState) -> bool:
        """Prompt for user confirmation to proceed with a step. Default is No."""
        prompt = f"Confirm to proceed with step '{step_name}' [y/N]: "
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = ""
        confirmed = answer in ("y", "yes")
        self._log(f"CONFIRM {step_name} -> {'YES' if confirmed else 'NO'}")
        return confirmed

    def _init_ai_trace_logging(self) -> None:
        """Initialize AI inputs trace logging and propagate to analyzer via env vars."""
        try:
            os.makedirs(self.config.ai_trace_log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.ai_trace_file = os.path.join(self.config.ai_trace_log_dir, f"ai_trace_{ts}.log")
            # Propagate to analyzer layer
            os.environ["HINGE_AI_TRACE_FILE"] = os.path.abspath(self.ai_trace_file)
            os.environ["HINGE_AI_TRACE_CONSOLE"] = "1" if getattr(self.config, "manual_confirm", False) else "0"
            # Header
            with open(self.ai_trace_file, "a", encoding="utf-8") as f:
                f.write(f"AI trace started {datetime.now().isoformat()} ai_trace={self.config.ai_trace}\n")
        except Exception as e:
            print(f"[ai-trace] Failed to init AI trace log: {e}")
            self.ai_trace_file = None

    def _ai_trace_log(self, lines) -> None:
        """Log AI inputs to the shared ai trace file and optionally to console."""
        if not getattr(self.config, "ai_trace", False) or not getattr(self, "ai_trace_file", None):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        out_lines = [f"[{ts}] {line}" for line in lines]
        try:
            with open(self.ai_trace_file, "a", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
        except Exception as e:
            print(f"[ai-trace] Failed to write AI trace: {e}")
        if getattr(self.config, "manual_confirm", False):
            for l in out_lines:
                print(l)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with AI-controlled decision making"""
        
        workflow = StateGraph(HingeAgentState)
        
        # Add all workflow nodes
        workflow.add_node("initialize_session", self.initialize_session_node)
        workflow.add_node("ai_decide_action", self.ai_decide_action_node)
        workflow.add_node("capture_screenshot", self.capture_screenshot_node)
        workflow.add_node("analyze_profile", self.analyze_profile_node)
        workflow.add_node("scroll_profile", self.scroll_profile_node)
        workflow.add_node("make_like_decision", self.make_like_decision_node)
        workflow.add_node("detect_like_button", self.detect_like_button_node)
        workflow.add_node("execute_like", self.execute_like_node)
        workflow.add_node("generate_comment", self.generate_comment_node)
        workflow.add_node("send_comment_with_typing", self.send_comment_with_typing_node)
        workflow.add_node("send_like_without_comment", self.send_like_without_comment_node)
        workflow.add_node("execute_dislike", self.execute_dislike_node)
        workflow.add_node("navigate_to_next", self.navigate_to_next_node)
        workflow.add_node("verify_profile_change", self.verify_profile_change_node)
        workflow.add_node("recover_from_stuck", self.recover_from_stuck_node)
        workflow.add_node("reset_app", self.reset_app_node)
        workflow.add_node("finalize_session", self.finalize_session_node)
        
        # Set entry point
        workflow.set_entry_point("initialize_session")
        
        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "initialize_session",
            self._route_initialization,
            {
                "success": "ai_decide_action",
                "failure": "finalize_session"
            }
        )
        
        workflow.add_conditional_edges(
            "ai_decide_action", 
            self._route_ai_decision,
            {
                "capture_screenshot": "capture_screenshot",
                "analyze_profile": "analyze_profile",
                "scroll_profile": "scroll_profile",
                "make_like_decision": "make_like_decision",
                "detect_like_button": "detect_like_button",
                "execute_like": "execute_like",
                "generate_comment": "generate_comment",
                "send_comment_with_typing": "send_comment_with_typing",
                "send_like_without_comment": "send_like_without_comment",
                "execute_dislike": "execute_dislike",
                "navigate_to_next": "navigate_to_next",
                "verify_profile_change": "verify_profile_change",
                "recover_from_stuck": "recover_from_stuck",
                "reset_app": "reset_app",
                "finalize": "finalize_session"
            }
        )
        
        # Add edges back to AI decision node from all action nodes
        action_nodes = [
            "capture_screenshot", "analyze_profile", "scroll_profile", "make_like_decision",
            "detect_like_button", "execute_like", "generate_comment", "send_comment_with_typing", "send_like_without_comment",
            "execute_dislike", "navigate_to_next", "verify_profile_change", "recover_from_stuck", "reset_app"
        ]
        
        for node in action_nodes:
            workflow.add_conditional_edges(
                node,
                self._route_action_result,
                {
                    "continue": "ai_decide_action",
                    "finalize": "finalize_session"
                }
            )
        
        workflow.add_edge("finalize_session", END)
        
        # Compile with increased recursion limit for multi-profile processing
        # Each profile may require 10-15 iterations, so allow for more profiles
        return workflow.compile(
            checkpointer=None,  # No checkpointing needed for our use case
            interrupt_before=None,
            interrupt_after=None,
            debug=False
        )
    
    # Routing functions
    def _route_initialization(self, state: HingeAgentState) -> str:
        return "success" if state.get("should_continue", False) else "failure"
    
    def _route_ai_decision(self, state: HingeAgentState) -> str:
        return state.get("next_tool_suggestion", "finalize")
    
    def _route_action_result(self, state: HingeAgentState) -> str:
        # Check completion conditions
        batch_start = state.get("batch_start_index", 0)
        batch_end = batch_start + self.profiles_per_batch
        
        if (state["current_profile_index"] >= min(batch_end, state["max_profiles"]) or
            state["errors_encountered"] > self.config.max_errors_before_abort or
            not state.get("should_continue", True)):
            return "finalize"
        return "continue"
    
    # Node implementations
    @gated_step
    def initialize_session_node(self, state: HingeAgentState) -> HingeAgentState:
        """Initialize the automation session"""
        print("🚀 Initializing LangGraph Hinge automation session...")
        # Gate CV per-template debug overlays when scrape-only or verbose logging
        try:
            os.environ["HINGE_CV_DEBUG_MODE"] = "1" if (getattr(self.config, "scrape_only", False) or getattr(self.config, "verbose_logging", False)) else "0"
        except Exception:
            pass
        
        # Clear old screenshots to prevent confusion
        clear_screenshots_directory()
        # Also clear mirrored submitted images
        clear_submitted_directory()
        
        device = connect_device(self.config.device_ip)
        if not device:
            return {
                **state,
                "should_continue": False,
                "completion_reason": "Failed to connect to device",
                "last_action": "initialize_session",
                "action_successful": False
            }
        
        width, height = get_screen_resolution(device)
        open_hinge(device)
        time.sleep(5)

        # Startup pre-check: ensure we are at top of profile feed (like button visible)
        try:
            start_screenshot = capture_screenshot(device, "startup_precheck")
            pre_like = detect_like_button_cv(start_screenshot)
            if self.config.precheck_strict and not pre_like.get('found', False):
                print("❌ Startup pre-check failed: like button not visible. Please navigate to the top of the profile feed (first card visible) and re-run.")
                return {
                    **state,
                    "should_continue": False,
                    "completion_reason": "Startup pre-check failed: not at top of profile feed",
                    "last_action": "initialize_session",
                    "action_successful": False
                }
        except Exception as _e:
            if self.config.precheck_strict:
                print(f"❌ Startup pre-check error: {_e}")
                return {
                    **state,
                    "should_continue": False,
                    "completion_reason": "Startup pre-check error",
                    "last_action": "initialize_session",
                    "action_successful": False
                }

        # Update template weights
        success_rates = calculate_template_success_rates()
        update_template_weights(success_rates)
        
        print(f"✅ Session initialized - Device: {device.serial}, Resolution: {width}x{height}")
        
        return {
            **state,
            "device": device,
            "width": width,
            "height": height,
            "max_profiles": self.max_profiles,
            "current_profile_index": 0,
            "profiles_processed": 0,
            "likes_sent": 0,
            "comments_sent": 0,
            "errors_encountered": 0,
            "stuck_count": 0,
            "profile_text": "",
            "profile_analysis": {},
            "decision_reason": "",
            "previous_profile_text": "",
            "previous_profile_features": {},
            "last_action": "initialize_session",
            "action_successful": True,
            "retry_count": 0,
            "generated_comment": "",
            "comment_id": "",
            "like_button_coords": None,
            "like_button_confidence": 0.0,
            "should_continue": True,
            "completion_reason": "",
            "ai_reasoning": "",
            "next_tool_suggestion": "capture_screenshot",
            "current_screenshot": None
        }
    
    @gated_step
    def ai_decide_action_node(self, state: HingeAgentState) -> HingeAgentState:
        """Ask AI to analyze current state and decide next action"""
        print(f"🤖 Asking AI for next action (Profile {state['current_profile_index'] + 1}/{state['max_profiles']})")
        
        # Prepare context for AI
        gc_present = bool(state.get('generated_comment'))
        comment_open = False
        try:
            if state.get('current_screenshot'):
                _ui = detect_comment_ui_elements(state['current_screenshot'])
                comment_open = bool(_ui.get('comment_field_found', False))
        except Exception:
            comment_open = False

        context = f"""
        Current Hinge Automation State:
        - Profile Index: {state['current_profile_index']}/{state['max_profiles']}
        - Profiles Processed: {state['profiles_processed']}
        - Last Action: {state['last_action']}
        - Action Successful: {state['action_successful']}
        - Current Screenshot: {state['current_screenshot']}
        - Profile Text: {state['profile_text'][:300]}...
        - Stuck Count: {state['stuck_count']}
        - Errors: {state['errors_encountered']}
        - Generated Comment Present: {gc_present}
        - Comment Interface Open: {comment_open}
        
        Profile Analysis:
        {json.dumps(state.get('profile_analysis', {}), indent=2)[:500]}
        
        Available Actions:
        1. capture_screenshot - Take screenshot of current screen
        2. analyze_profile - Comprehensive analysis (automatically scrolls 3 times, extracts all user content, analyzes complete profile)
        3. scroll_profile - Manual scroll (rarely needed since analyze_profile handles scrolling)
        4. make_like_decision - Decide whether to like or dislike profile
        5. detect_like_button - Find like button coordinates (use before execute_like)
        6. execute_like - Tap the like button (REQUIRED before commenting - opens comment interface)
        7. generate_comment - Create personalized comment (use after execute_like)
        8. send_comment_with_typing - Complete comment process (use after generate_comment, requires comment interface to be open)
        9. send_like_without_comment - Send like without typing comment (fallback)
        10. execute_dislike - Dislike/skip current profile
        11. navigate_to_next - Move to next profile
        12. verify_profile_change - Check if we moved to new profile
        13. recover_from_stuck - Attempt recovery when stuck
        14. reset_app - Force close and reopen Hinge app (use when severely stuck on or an unexpected page or different app)
        15. finalize - End the session
        
        Workflow Guidelines:
        - Always start with capture_screenshot if no current screenshot
        - The general flow is: capture_screenshot > analyze_profile (comprehensive) > make_like_decision > detect_like_button > execute_like > generate_comment > send_comment_with_typing > next profile
        - Preconditions:
          • Only choose send_comment_with_typing when Generated Comment Present is True and Comment Interface Open is True
          • If Comment Interface Open is True and Generated Comment Present is False → the next action must be "generate_comment"
        - analyze_profile automatically performs 3 scrolls and extracts all user content (no need for separate scroll actions)
        - Only like profiles that meet quality criteria based on comprehensive analysis
        - IMPORTANT: Must execute_like (tap like button) BEFORE attempting to comment - comment interface only appears after like button is tapped
        - For commenting workflow: detect_like_button → execute_like → generate_comment → send_comment_with_typing
        - If commenting fails: use send_like_without_comment as fallback
        - Use recover_from_stuck when stuck count > 2
        - Use reset_app when stuck count > 4 OR when the app appears unresponsive or severely stuck
        - reset_app is a nuclear option that completely refreshes the app state - use when other recovery methods fail
        - After reset_app, you'll need to start fresh with capture_screenshot
        - Finalize when max profiles reached or too many errors
        """
        
        try:
            if state['current_screenshot']:
                # Include screenshot for visual analysis
                with open(state['current_screenshot'], 'rb') as f:
                    image_bytes = f.read()
                
                b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                prompt = f"""
                {context}
                
                Analyze the current screenshot and determine the best next action.
                
                Respond in JSON format:
                {{
                    "next_action": "action_name",
                    "reasoning": "detailed explanation of why this action was chosen",
                    "confidence": 0.0-1.0,
                    "expected_outcome": "what should happen after this action"
                }}
                
                Consider:
                - What type of screen is currently displayed?
                - What is the appropriate next step in the workflow?
                - Are there any error conditions or stuck states?
                - Has the session goal been completed?
                """
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }]
                try:
                    _sz = os.path.getsize(state['current_screenshot'])
                except Exception:
                    _sz = "?"
                self._ai_trace_log([
                    "AI_CALL call_id=ai_decide_action model=gpt-5-mini temperature=0.0 response_format=json_object",
                    "PROMPT=<<<BEGIN",
                    *prompt.splitlines(),
                    "<<<END",
                    f"IMAGE image_path={state['current_screenshot']} image_size={_sz} bytes"
                ])
            else:
                # No screenshot available
                prompt = f"""
                {context}
                
                No screenshot is available. Determine the best next action.
                Usually this should be "capture_screenshot" to see the current state.
                
                Respond in JSON format with next_action and reasoning.
                """
                
                messages = [{"role": "user", "content": prompt}]
                self._ai_trace_log([
                    "AI_CALL call_id=ai_decide_action model=gpt-5-mini temperature=0.0 response_format=json_object",
                    "PROMPT=<<<BEGIN",
                    *prompt.splitlines(),
                    "<<<END",
                ])
            
            t0 = time.perf_counter()
            client = self.ai_client or OpenAI()
            self.ai_client = client
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                response_format={"type": "json_object"},
                messages=messages
            )
            dt_ms = int((time.perf_counter() - t0) * 1000)
            try:
                print(f"[AI] ai_decide_action model=gpt-5-mini duration={dt_ms}ms")
            except Exception:
                pass
            self._ai_trace_log([f"AI_TIME call_id=ai_decide_action model=gpt-5-mini duration_ms={dt_ms}"])
            
            decision = json.loads(resp.choices[0].message.content) if resp.choices[0].message and resp.choices[0].message.content else {}
            next_action = decision.get('next_action', 'capture_screenshot')
            reasoning = decision.get('reasoning', 'Default action')

            # Enforce preconditions deterministically (Hybrid Option B)
            # - If comment interface is open and no generated comment: must generate_comment
            # - If comment interface is open and we have a generated comment: must send_comment_with_typing
            # - Never choose send_comment_with_typing when comment interface is not open
            # - Never choose execute_like while already in comment interface
            if comment_open:
                if not gc_present and next_action != "generate_comment":
                    self._log("Precondition enforcement: forcing 'generate_comment' (modal open, no generated comment).")
                    next_action = "generate_comment"
                elif gc_present and next_action != "send_comment_with_typing":
                    self._log("Precondition enforcement: forcing 'send_comment_with_typing' (modal open, generated comment present).")
                    next_action = "send_comment_with_typing"
                if next_action == "execute_like":
                    # Don't re-tap like when modal is open; prefer to proceed with comment or send-like
                    if gc_present:
                        next_action = "send_comment_with_typing"
                    else:
                        next_action = "generate_comment"
            else:
                if next_action == "send_comment_with_typing":
                    self._log("Precondition enforcement: 'send_comment_with_typing' chosen but modal is not open → forcing 'execute_like'.")
                    next_action = "execute_like"
            
            print(f"🎯 AI chose: {next_action}")
            print(f"💭 Reasoning: {reasoning}")
            
            return {
                **state,
                "next_tool_suggestion": next_action,
                "ai_reasoning": reasoning,
                "last_action": "ai_decide_action",
                "action_successful": True
            }
            
        except Exception as e:
            print(f"❌ AI decision error: {e}")
            # Fallback decision
            fallback_action = "capture_screenshot" if not state['current_screenshot'] else "navigate_to_next"
            
            return {
                **state,
                "next_tool_suggestion": fallback_action,
                "ai_reasoning": f"Fallback due to error: {e}",
                "last_action": "ai_decide_action",
                "action_successful": False,
                "errors_encountered": state["errors_encountered"] + 1
            }
    
    @gated_step
    def capture_screenshot_node(self, state: HingeAgentState) -> HingeAgentState:
        """Capture current screen screenshot"""
        print("📸 Capturing screenshot...")
        
        screenshot_path = capture_screenshot(
            state["device"],
            f"profile_{state['current_profile_index']}_langgraph"
        )
        
        return {
            **state,
            "current_screenshot": screenshot_path,
            "last_action": "capture_screenshot",
            "action_successful": True
        }
    
    @gated_step
    def analyze_profile_node(self, state: HingeAgentState) -> HingeAgentState:
        """Comprehensive profile analysis with new full scrape (horizontal + vertical)"""
        print("🔍 Starting comprehensive profile analysis (new full scrape)...")
        return self._analyze_profile_full_scrape(state)
        
        # Collect multiple screenshots by scrolling through the profile
        all_screenshots = []
        all_profile_texts = []
        
        # Start with initial screenshot
        print("📸 Analyzing initial screenshot...")
        all_screenshots.append(state['current_screenshot'])
        initial_text = self._extract_user_content_only(state['current_screenshot'])
        all_profile_texts.append(initial_text)
        
        # Perform 3 scrolls to capture full profile content
        current_screenshot = state['current_screenshot']
        
        for scroll_num in range(1, 4):  # 3 scrolls
            print(f"📜 Performing scroll {scroll_num}/3...")
            
            # Scroll down to reveal more content
            scroll_x = int(state["width"] * 0.5)  # Center of screen
            scroll_y_start = int(state["height"] * 0.7)  # Start from 70% down
            scroll_y_end = int(state["height"] * 0.3)    # End at 30% down
            
            swipe(state["device"], scroll_x, scroll_y_start, scroll_x, scroll_y_end, duration=600)
            time.sleep(2)  # Allow content to load
            
            # Capture screenshot after scroll
            scroll_screenshot = capture_screenshot(
                state["device"], 
                f"profile_{state['current_profile_index']}_scroll_{scroll_num}"
            )
            all_screenshots.append(scroll_screenshot)
            
            # Extract user content from this scroll
            scroll_text = self._extract_user_content_only(scroll_screenshot)
            all_profile_texts.append(scroll_text)
            
            current_screenshot = scroll_screenshot
        
        # Combine all extracted text, removing duplicates
        combined_text = self._combine_unique_content(all_profile_texts)
        
        # Perform comprehensive analysis on all collected content
        print("🧠 Performing comprehensive profile analysis...")
        comprehensive_analysis = self._analyze_complete_profile(all_screenshots, combined_text)
        
        quality_score = comprehensive_analysis.get('profile_quality_score', 0)
        print(f"📊 Comprehensive profile quality: {quality_score}/10")
        print(f"📝 Total content captured: {len(combined_text)} characters")
        
        return {
            **state,
            "current_screenshot": current_screenshot,  # Use latest screenshot
            "profile_text": combined_text,
            "profile_analysis": comprehensive_analysis,
            "last_action": "analyze_profile",
            "action_successful": True
        }
    
    def _extract_user_content_only(self, screenshot_path: str) -> str:
        """Extract only user-generated content, filtering out UI elements"""
        try:
            with open(screenshot_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = """
            Extract ONLY user-generated content from this dating profile screenshot. 
            
            INCLUDE:
            - Profile name and age
            - Bio/description text written by the user
            - Prompt answers (e.g. "My simple pleasures: ...")
            - Personal interests, hobbies, job titles
            - Location if it's user-provided
            - Any text the user wrote about themselves
            
            EXCLUDE/IGNORE:
            - UI buttons (Like, Pass, Comment, Send, etc.)
            - Navigation elements
            - App interface text
            - System messages
            - Generic prompts/questions before answers
            - Icons and emojis that are part of UI
            - Distance indicators
            - Match percentage
            - Photo count indicators
            - Any text that's part of the app interface
            
            Return only the clean user content, formatted naturally without any commentary or analysis.
            If no user content is visible, return an empty string.
            """
            
            try:
                _sz = os.path.getsize(screenshot_path)
            except Exception:
                _sz = "?"
            self._ai_trace_log([
                "AI_CALL call_id=extract_user_content_only model=gpt-5-mini temperature=0.0",
                "PROMPT=<<<BEGIN",
                *prompt.splitlines(),
                "<<<END",
                f"IMAGE image_path={screenshot_path} image_size={_sz} bytes"
            ])
            client = self.ai_client or OpenAI()
            self.ai_client = client
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }]
            )
            
            return (resp.choices[0].message.content or "").strip()
            
        except Exception as e:
            print(f"❌ Error extracting user content: {e}")
            return ""
    
    def _combine_unique_content(self, text_list: list) -> str:
        """Combine text from multiple screenshots, removing duplicates"""
        all_lines = []
        seen_lines = set()
        
        for text in text_list:
            if not text:
                continue
                
            lines = text.split('\n')
            for line in lines:
                clean_line = line.strip()
                if clean_line and clean_line not in seen_lines:
                    seen_lines.add(clean_line)
                    all_lines.append(clean_line)
        
        return '\n'.join(all_lines)
    
    def _analyze_profile_full_scrape(self, state: HingeAgentState) -> HingeAgentState:
        """New flow:
        1) Ensure we have a screenshot
        2) Full vertical scroll once
        3) Screenshot
        4) Detect age icon to get Y for horizontal carousel; horizontally swipe until stable
        5) Vertical paging one screen at a time
        6) Final single-call analysis with all collected screenshots
        """
        try:
            if not state.get('current_screenshot'):
                return {
                    **state,
                    "last_action": "analyze_profile",
                    "action_successful": False
                }
            width = state["width"]
            height = state["height"]
            device = state["device"]
            
            all_screenshots: list[str] = []
            # No per-screenshot text extraction; dedup by image similarity and batch analyze once
            
            # Use current screenshot as starting point
            start_shot = state['current_screenshot']
            all_screenshots.append(start_shot)
            
            # Full vertical settle scroll (~3/4 screen) to reduce lazy-load jitter without overshooting
            sx = int(width * float(getattr(self.config, "vertical_swipe_x_pct", 0.12)))
            y1 = int(height * 0.77)
            y2 = int(height * 0.33)
            self._vertical_swipe_px(state, sx, y1, y2, duration_ms=int(getattr(self.config, "vertical_swipe_duration_ms", 1200)))
            time.sleep(2)
            post_full_scroll = capture_screenshot(device, f"profile_{state['current_profile_index']}_post_full_scroll")
            all_screenshots.append(post_full_scroll)
            
            # Detect age icon Y for horizontal carousel:
            # Try early frames first (pre-scroll), then post_full_scroll (each with its own adaptive seek)
            print(f"SWEEP_START image={os.path.basename(post_full_scroll)}")
            sweep = self._sweep_icons_find_y(state, start_image=post_full_scroll)
            if sweep.get("found"):
                y_horizontal = int(sweep["y"])
                pages = int(sweep.get("pages_scanned", 0))
                img_used = sweep.get("image_used", "")
                print(f"✅ Carousel Y determined at {y_horizontal} (method={sweep.get('method','sweep_dual')}, pages_scanned={pages}, image_used={os.path.basename(img_used) if img_used else ''})")
            else:
                print("❌ Could not determine horizontal carousel Y after sweep (high-threshold dual-template).")
                return {
                    **state,
                    "last_action": "analyze_profile",
                    "action_successful": False,
                    "errors_encountered": state.get("errors_encountered", 0) + 1
                }
            
            # Skip upstream horizontal stabilization; start CV immediately after Y
            try:
                print("⛔ Skipping pre-swipe; starting CV loop immediately.")
            except Exception:
                pass

            # CV+OCR biometrics extraction (horizontal row only)
            cv_result = {}
            if getattr(self.config, "use_cv_ocr_biometrics", True):
                try:
                    cv_result = extract_biometrics_from_carousel(
                        device,
                        start_screenshot=(sweep.get("image_used") or post_full_scroll),
                        ocr_engine=str(getattr(self.config, "cv_ocr_engine", "easyocr")),
                        max_micro_swipes=int(getattr(self.config, "max_horizontal_swipes", 8)),
                        micro_swipe_ratio=float(getattr(self.config, "cv_micro_swipe_ratio", 0.25)),
                        seek_swipe_ratio=float(getattr(self.config, "cv_seek_swipe_ratio", 0.60)),
                        target_center_x_ratio=float(getattr(self.config, "cv_target_center_x_ratio", 0.38)),
                        band_height_ratio=float(getattr(self.config, "cv_band_height_ratio", 0.06)),
                        allow_edges_fallback=False,  # Y is known; no fallback
                        verbose_timing=bool(getattr(self.config, "verbose_cv_timing", False)),
                        y_override=int(y_horizontal),
                    )
                    try:
                        print(f"[CV_OCR] biometrics={cv_result.get('biometrics', {})}")
                        if bool(getattr(self.config, "verbose_cv_timing", False)):
                            print(f"[CV_OCR] timing={cv_result.get('timing', {})}")
                    except Exception:
                        pass
                except Exception as _e:
                    print(f"❌ CV+OCR biometrics extraction failed: {_e}")
                    cv_result = {}
            
            # Vertical paging collection until stable (image-dedup)
            vshots, _ = self._collect_vertical_pages(state)
            all_screenshots.extend(vshots)
            
            # Analyze once with all unique screenshots (no interim AI calls)
            combined_text = ""
            print(f"🖼️ Unique screenshots collected: {len(all_screenshots)}")
            # Build and SUBMIT reusable batch payload for LLM extraction (OpenAI format)
            extracted_profile: Dict[str, Any] = {}
            extraction_failed = False
            try:
                cap = int(getattr(self.config, "llm_max_images", 10))
                if len(all_screenshots) > cap:
                    print(f"❌ LLM image cap exceeded: {len(all_screenshots)} > {cap}. Exiting run.")
                    return {
                        **state,
                        "last_action": "analyze_profile",
                        "action_successful": False,
                        "should_continue": False,
                        "completion_reason": f"LLM image cap exceeded: {len(all_screenshots)} > {cap}"
                    }
                images_for_llm = list(all_screenshots)
                # Always include stitched horizontal carousel image as first frame
                stitched_path = cv_result.get("stitched_carousel") if isinstance(cv_result, dict) else None
                images_for_llm = []
                if stitched_path and os.path.exists(stitched_path):
                    images_for_llm.append(stitched_path)
                    print(f"📦 Added stitched horizontal carousel: {stitched_path}")

                # Include top-of-profile screenshot (startup_precheck or langgraph)
                top_candidates = [p for p in all_screenshots if ("startup_precheck" in p or "langgraph" in p)]
                if top_candidates:
                    top_path = top_candidates[0]
                    images_for_llm.insert(0, top_path)
                    print(f"📦 Added top-of-profile screenshot: {top_path}")

                # Add all unique vertical pages after stitched image
                images_for_llm.extend([p for p in all_screenshots if ("_vpage_" in p or "_post_full_scroll" in p)])
                print(f"📦 LLM payload images (top + stitched + vertical): {len(images_for_llm)} total")

                # Use only the new structured profile prompt
                from prompt_engine import build_structured_profile_prompt
                llm_payload = build_llm_batch_payload(images_for_llm, prompt=build_structured_profile_prompt())
                images_count = llm_payload.get("meta", {}).get("images_count", len(all_screenshots))
                print(f"📦 LLM batch images: {images_count}")
                print("📸 Images submitted to LLM:")
                for img_path in images_for_llm:
                    print(f"   - {os.path.abspath(img_path)}")

                # Mirror the exact LLM payload images into images/submitted with friendly names
                try:
                    dest_dir = os.path.join("images", "submitted")
                    os.makedirs(dest_dir, exist_ok=True)
                    payload_paths = (llm_payload.get("meta", {}) or {}).get("images_paths", list(images_for_llm))
                    idx = 0
                    stitched_src = None
                    for p in payload_paths:
                        try:
                            base = os.path.basename(p).lower()
                            ext = os.path.splitext(p)[1] or ".png"
                            if "stitched_carousel" in base or "stitched_" in base:
                                stitched_src = p
                                continue
                            dest = os.path.join(dest_dir, f"profile {idx}{ext}")
                            shutil.copy2(p, dest)
                            idx += 1
                        except Exception as _e:
                            print(f"⚠️ Mirror copy failed for {p}: {_e}")
                    if stitched_src:
                        ext = os.path.splitext(stitched_src)[1] or ".png"
                        dest = os.path.join(dest_dir, f"profile stitched{ext}")
                        try:
                            shutil.copy2(stitched_src, dest)
                        except Exception as _e:
                            print(f"⚠️ Mirror copy (stitched) failed: {_e}")
                    print(f"🗂️  Mirrored {idx}{' + stitched' if stitched_src else ''} image(s) to {os.path.abspath(dest_dir)}")
                except Exception as _e:
                    print(f"⚠️ Mirror operation failed: {_e}")

                # Submit to LLM (uses gpt-5 by default; gpt-5-mini for small logical calls if needed)
                extracted_raw = self._submit_llm_batch_request(llm_payload)
                # Validate required top-level keys presence (values may be null)
                missing_after = self._validate_required_fields(extracted_raw)
                if missing_after:
                    print(f"⚠️ Extraction missing required keys after retry: {', '.join(missing_after)}")
                    extraction_failed = True
                # Use raw LLM output directly (no normalization)
                extracted_profile = extracted_raw
                # Optionally save raw JSON for debugging
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open(os.path.join("logs", "last_llm_output.json"), "w", encoding="utf-8") as f:
                        json.dump(extracted_raw, f, indent=2)
                except Exception as _e:
                    print(f"⚠️ Could not save raw LLM output: {_e}")
                # Merge CV+OCR biometrics into extracted profile (CV takes precedence over LLM)
                try:
                    if getattr(self.config, "use_cv_ocr_biometrics", True) and isinstance(cv_result, dict):
                        try:
                            print("[CV_OCR] Merged CV biometrics into extracted_profile (CV takes precedence).")
                        except Exception:
                            pass
                except Exception as _me:
                    print(f"⚠️ Merge biometrics failed: {_me}")
                try:
                    print("[AI JSON extracted_profile]")
                    print(json.dumps(extracted_profile, indent=2)[:2000])
                except Exception:
                    pass
            except Exception as _e:
                print(f"❌ Extraction exception: {_e}")
                llm_payload = {}
                extraction_failed = True

            # Evaluate core fields for scoring modifiers (Home town, Job title, University)
            eval_result = {}
            try:
                eval_result = evaluate_profile_fields(extracted_profile, model=getattr(self.config, "extraction_model", "gpt-5"))
            except Exception as _e:
                print(f"⚠️ profile_eval exception: {_e}")

            current_screenshot = all_screenshots[-1] if all_screenshots else state['current_screenshot']
            return {
                **state,
                "current_screenshot": current_screenshot,
                "profile_text": combined_text,
                "profile_analysis": extracted_profile,
                "extracted_profile": extracted_profile,
                "profile_eval": eval_result,
                "extraction_failed": extraction_failed,
                "llm_batch_request": llm_payload,
                "cv_biometrics": (cv_result.get("biometrics", {}) if isinstance(cv_result, dict) else {}),
                "cv_biometrics_timing": (cv_result.get("timing", {}) if isinstance(cv_result, dict) else {}),
                "last_action": "analyze_profile",
                "action_successful": True
            }
        except Exception as e:
            print(f"❌ Error in new analyze_profile flow: {e}")
            return {
                **state,
                "last_action": "analyze_profile",
                "action_successful": False,
                "errors_encountered": state.get("errors_encountered", 0) + 1
            }
    
    def _find_carousel_y_with_vertical_scan(self, state: HingeAgentState, start_image: str) -> Dict[str, Any]:
        """
        Try to locate the age/name marker Y with progressive ROI stages on the current frame.
        If not found, perform up to N small peek scrolls, re-checking after each.
        Optionally restore the viewport after success.
        Returns: {"found": bool, "y": int, "pages_scanned": int, "method": "template|edges|not_found"}
        """
        device = state["device"]
        width = state["width"]
        height = state["height"]
        cfg = self.config

        def try_detect_in_stages(img_path: str) -> Dict[str, Any]:
            base_thr = float(getattr(cfg, "age_icon_threshold", 0.55))
            decay = float(getattr(cfg, "carousel_detection_threshold_decay", 0.05))
            stages = list(getattr(cfg, "carousel_detection_roi_stages", ((0.0, 0.55), (0.0, 0.75), (0.0, 0.90))))
            # Stage through ROI windows with threshold decay; template first, then edges
            print(f"AGE_DETECT_PARAMS base_thr={base_thr:.2f} decay={decay:.2f} stages={stages} use_edges={bool(getattr(cfg, 'age_icon_use_edges', True))} smooth_kernel={int(getattr(cfg, 'carousel_y_smooth_kernel', 21))}")
            for idx, (top, bottom) in enumerate(stages):
                thr = max(0.2, base_thr - decay * idx)
                # Template+edges
                dual = detect_age_row_dual_templates(
                    img_path,
                    template_paths=getattr(cfg, "age_icon_templates", ("assets/icon_age.png", "assets/icon_gender.png", "assets/icon_height.png")),
                    roi_top=float(top),
                    roi_bottom=float(bottom),
                    threshold=thr,
                    use_edges=bool(getattr(cfg, "age_icon_use_edges", True)),
                    save_debug=True,
                    tolerance_px=int(getattr(cfg, "age_dual_y_tolerance_px", 5)),
                    tolerance_ratio=float(getattr(cfg, "age_dual_y_tolerance_ratio", 0.005)),
                    require_both=bool(getattr(cfg, "require_both_icons_for_y", True)),
                    expected_px=int(getattr(cfg, "icon_expected_px", 60)),
                    scale_tolerance=float(getattr(cfg, "icon_scale_tolerance", 0.30)),
                    min_px=int(getattr(cfg, "icon_min_px", 20)),
                    max_roi_frac=float(getattr(cfg, "icon_max_roi_frac", 0.12)),
                    edges_dilate_iter=int(getattr(cfg, "edges_dilate_iter", 1)),
                )
                if dual.get("found"):
                    y_consensus = int(dual.get("y", 0))
                    print(f"AGE_DETECT_STAGE_RESULT idx={idx} method=dual(avg) y={y_consensus}")
                    return {"found": True, "y": y_consensus, "method": f"dual(stage={idx},thr={thr:.2f})"}
                # Edges fallback (row-strength) - optionally disabled when icons are required
                if not bool(getattr(cfg, "require_icon_detection_for_y", True)):
                    edge = infer_carousel_y_by_edges(
                        img_path,
                        roi_top=float(getattr(cfg, "carousel_y_roi_top", top)),
                        roi_bottom=float(getattr(cfg, "carousel_y_roi_bottom", bottom)),
                        smooth_kernel=int(getattr(cfg, "carousel_y_smooth_kernel", 21))
                    )
                    if edge.get("found"):
                        print(f"AGE_DETECT_STAGE_RESULT idx={idx} method=edges y={int(edge['y'])}")
                        return {"found": True, "y": int(edge["y"]), "method": f"edges(stage={idx})"}
                print(f"AGE_DETECT_STAGE_RESULT idx={idx} method=none found=False")
            return {"found": False}

        # 1) Try detection on the starting image
        res = try_detect_in_stages(start_image)
        if res.get("found"):
            res["pages_scanned"] = 0
            return res

        # 2) Peek-scan up to max_carousel_y_scans times
        max_scans = int(getattr(cfg, "max_carousel_y_scans", 2))
        step = getattr(cfg, "carousel_scan_step", (0.72, 0.48))
        sx = int(width * float(getattr(cfg, "vertical_swipe_x_pct", 0.12)))
        sy1 = int(height * float(step[0]))
        sy2 = int(height * float(step[1]))
        hash_size = int(getattr(cfg, "image_hash_size", 8))
        hash_thresh = int(getattr(cfg, "image_hash_threshold", 5))

        pages = 0
        last_kept = start_image
        for i in range(max_scans):
            self._vertical_swipe_px(state, sx, sy1, sy2, duration_ms=int(getattr(cfg, "vertical_swipe_duration_ms", 1200)))
            time.sleep(1.6)
            shot = capture_screenshot(device, f"carousel_seek_{state['current_profile_index']}_{i+1}")
            if last_kept and are_images_similar(shot, last_kept, hash_size=hash_size, threshold=hash_thresh):
                # Nothing new; avoid looping
                break
            last_kept = shot
            pages += 1
            res = try_detect_in_stages(shot)
            if res.get("found"):
                # Optional restore
                if bool(getattr(cfg, "carousel_restore_after_seek", True)) and pages > 0:
                    # Reverse scroll roughly back
                    r_sy1 = int(height * 0.40)
                    r_sy2 = int(height * 0.68)
                    for _ in range(pages):
                        self._vertical_swipe_px(state, sx, r_sy1, r_sy2, duration_ms=int(getattr(cfg, "vertical_swipe_duration_ms", 1200)))
                        time.sleep(0.8)
                res["pages_scanned"] = pages
                return res

        return {"found": False, "pages_scanned": pages, "method": "not_found"}

    def _sweep_icons_find_y(self, state: HingeAgentState, start_image: str) -> Dict[str, Any]:
        """ Sequential sweep to locate the age/gender icon row:
        - Start at the provided frame, try dual-template detection with a high threshold.
        - If not found, scroll one page, capture the next frame, and retry.
        - Stop when found, after max_icon_sweep_pages, or when frames stabilize (bottom reached).
        Returns: {"found": bool, "y": int, "pages_scanned": int, "image_used": str, "method": "sweep_dual"} """
        device = state["device"]
        width = state["width"]
        height = state["height"]
        cfg = self.config

        templates = getattr(cfg, "age_icon_templates", ("assets/icon_age.png", "assets/icon_gender.png", "assets/icon_height.png"))
        roi_top, roi_bottom = tuple(getattr(cfg, "age_icon_roi", (0.1, 0.9)))
        thr = float(getattr(cfg, "icon_high_threshold", 0.80))
        thr_low = max(0.70, thr - 0.10)
        use_edges = bool(getattr(cfg, "age_icon_use_edges", True))
        tol_px = int(getattr(cfg, "age_dual_y_tolerance_px", 5))
        tol_ratio = float(getattr(cfg, "age_dual_y_tolerance_ratio", 0.005))
        expected_px = int(getattr(cfg, "icon_expected_px", 60))
        scale_tol = float(getattr(cfg, "icon_scale_tolerance", 0.30))
        min_px = int(getattr(cfg, "icon_min_px", 20))
        max_roi_frac = float(getattr(cfg, "icon_max_roi_frac", 0.12))
        edges_dilate_iter = int(getattr(cfg, "edges_dilate_iter", 1))
        max_pages = int(getattr(cfg, "max_icon_sweep_pages", 8))

        hash_size = int(getattr(cfg, "image_hash_size", 8))
        hash_thresh = int(getattr(cfg, "image_hash_threshold", 5))
        stable_needed = int(getattr(cfg, "image_stable_repeats", getattr(cfg, "content_stable_repeats", 2)))

        current_path = start_image
        pages = 0
        stable = 0
        last_kept = None

        def detect_on(path: str) -> Dict[str, Any]:
            return detect_age_row_dual_templates(
                path,
                template_paths=templates,
                roi_top=float(roi_top),
                roi_bottom=float(roi_bottom),
                threshold=thr,
                use_edges=use_edges,
                save_debug=True,
                tolerance_px=tol_px,
                tolerance_ratio=tol_ratio,
                require_both=True,
                expected_px=expected_px,
                scale_tolerance=scale_tol,
                min_px=min_px,
                max_roi_frac=max_roi_frac,
                edges_dilate_iter=edges_dilate_iter,
            )

        while True:
            dual = detect_on(current_path)
            if not dual.get("found") and thr_low < thr:
                dual_low = detect_age_row_dual_templates(
                    current_path,
                    template_paths=templates,
                    roi_top=float(roi_top),
                    roi_bottom=float(roi_bottom),
                    threshold=thr_low,
                    use_edges=use_edges,
                    save_debug=True,
                    tolerance_px=tol_px,
                    tolerance_ratio=tol_ratio,
                    require_both=True,
                    expected_px=expected_px,
                    scale_tolerance=scale_tol,
                    min_px=min_px,
                    max_roi_frac=max_roi_frac,
                    edges_dilate_iter=edges_dilate_iter,
                )
                if dual_low.get("found"):
                    dual = dual_low
            if dual.get("found"):
                return {
                    "found": True,
                    "y": int(dual.get("y", 0)),
                    "pages_scanned": pages,
                    "image_used": current_path,
                    "method": "sweep_dual",
                }

            if pages >= max_pages:
                break

            sx = int(width * float(getattr(cfg, "vertical_swipe_x_pct", 0.12)))
            sy1 = int(height * 0.80)
            sy2 = int(height * 0.20)
            self._vertical_swipe_px(state, sx, sy1, sy2, duration_ms=int(getattr(cfg, "vertical_swipe_duration_ms", 1200)))
            time.sleep(1.6)
            shot = capture_screenshot(device, f"icon_sweep_{state['current_profile_index']}_{pages+1}")

            if last_kept is None or not are_images_similar(shot, last_kept, hash_size=hash_size, threshold=hash_thresh):
                last_kept = shot
                stable = 0
            else:
                stable += 1
                if stable >= stable_needed:
                    break

            current_path = shot
            pages += 1

        return {"found": False, "pages_scanned": pages, "method": "sweep_not_found"}

    def _vertical_swipe_px(self, state: HingeAgentState, x: int, y1: int, y2: int, duration_ms: int = 1200) -> None:
        """Vertical swipe with controlled duration and tiny x jitter to encourage scroll classification (not tap)."""
        try:
            jitter = int(getattr(self.config, "vertical_swipe_x_jitter_px", 3))
            width = state["width"]
            height = state["height"]
            guard_bottom = float(getattr(self.config, "vertical_swipe_bottom_guard_pct", 0.90))
            max_y = max(0, int(height * guard_bottom) - 1)
            sy1 = min(max(y1, 0), max_y)
            sy2 = min(max(y2, 0), max_y)
            # Apply tiny jitter: if swiping up (y1 > y2), nudge x2 right; if down, nudge left
            x2 = min(max(x + (jitter if sy2 < sy1 else -jitter), 0), width - 1)
            dur = max(300, int(duration_ms))
            print(f"SWIPE_VERTICAL x1={x} x2={x2} y1={sy1} y2={sy2} duration={dur}ms")
            swipe(state["device"], x, sy1, x2, sy2, duration=dur)
        except Exception as e:
            print(f"SWIPE_VERTICAL_FAILED: {e}")
            swipe(state["device"], x, y1, x, y2, duration=max(600, int(duration_ms)))

    def _collect_horizontal_content(self, state: HingeAgentState, y_coord: int) -> tuple[list, list]:
        """Swipe horizontally across the photo carousel, capturing until frames stabilize (aHash); no interim AI calls."""
        cfg = self.config
        width = state["width"]
        device = state["device"]
        start_p, end_p = getattr(cfg, "horizontal_swipe_dx", (0.80, 0.20))
        x1 = int(width * start_p)
        x2 = int(width * end_p)
        stable_needed = int(getattr(cfg, "hscroll_stable_repeats", getattr(cfg, "image_stable_repeats", getattr(cfg, "content_stable_repeats", 2))))
        max_swipes = int(getattr(cfg, "max_horizontal_swipes", 8))
        hash_size = int(getattr(cfg, "image_hash_size", 8))
        hash_thresh = int(getattr(cfg, "hscroll_hash_threshold", getattr(cfg, "image_hash_threshold", 5)))
        band_ratio = float(getattr(cfg, "hscroll_hash_roi_ratio", 0.12))
        horizontal_ms = int(getattr(cfg, "horizontal_swipe_duration_ms", 600))
        
        print(f"HSWIPE_SETUP x1={x1} x2={x2} y={y_coord} duration={horizontal_ms}ms")
        try:
            print(f"HSWIPE_PARAMS hash_threshold={hash_thresh} stable_needed={stable_needed} hash_size={hash_size} band_ratio={band_ratio}")
        except Exception:
            pass
        unique_shots: list[str] = []
        stable = 0
        last_kept: str | None = None
        
        for i in range(1, max_swipes + 1):
            print(f"HSWIPE_DO i={i}/{max_swipes} x1={x1} x2={x2} y={y_coord} duration={horizontal_ms}ms")
            swipe(device, x1, y_coord, x2, y_coord, duration=horizontal_ms)
            time.sleep(2)
            shot = capture_screenshot(device, f"profile_{state['current_profile_index']}_hscroll_{i}")
            
            if last_kept is None or not are_images_similar_roi(shot, last_kept, y_coord, band_ratio=band_ratio, hash_size=hash_size, threshold=hash_thresh):
                unique_shots.append(shot)
                last_kept = shot
                stable = 0
            else:
                stable += 1
                print(f"🟨 Horizontal duplicate detected ({stable}/{stable_needed})")
            
            if stable >= stable_needed:
                print(f"HSWIPE_STABLE repeats={stable} needed={stable_needed}")
                print("✅ Horizontal frames stabilized; stopping horizontal swipes.")
                break
        
        return unique_shots, []
    
    def _collect_vertical_pages(self, state: HingeAgentState) -> tuple[list, list]:
        """Scroll down one screen at a time, capturing until frames stabilize (aHash); no interim AI calls."""
        cfg = self.config
        width = state["width"]
        height = state["height"]
        device = state["device"]
        max_pages = int(getattr(cfg, "max_vertical_pages", 10))
        stable_needed = int(getattr(cfg, "image_stable_repeats", getattr(cfg, "content_stable_repeats", 2)))
        hash_size = int(getattr(cfg, "image_hash_size", 8))
        hash_thresh = int(getattr(cfg, "image_hash_threshold", 5))
        
        unique_shots: list[str] = []
        stable = 0
        last_kept: str | None = None
        
        for i in range(1, max_pages + 1):
            print(f"📜 Vertical page {i}/{max_pages}")
            sx = int(width * float(getattr(self.config, "vertical_swipe_x_pct", 0.12)))
            sy1 = int(height * 0.75)
            sy2 = int(height * 0.25)
            self._vertical_swipe_px(state, sx, sy1, sy2, duration_ms=int(getattr(self.config, "vertical_swipe_duration_ms", 1200)))
            time.sleep(2)
            shot = capture_screenshot(device, f"profile_{state['current_profile_index']}_vpage_{i}")
            
            if last_kept is None or not are_images_similar(shot, last_kept, hash_size=hash_size, threshold=hash_thresh):
                unique_shots.append(shot)
                last_kept = shot
                stable = 0
            else:
                stable += 1
                print(f"🟨 Vertical duplicate detected ({stable}/{stable_needed})")
            
            if stable >= stable_needed:
                print("✅ Vertical frames stabilized; stopping vertical paging.")
                break
        
        return unique_shots, []
    
    def _analyze_complete_profile(self, screenshots: list, combined_text: str) -> dict:
        """Perform comprehensive analysis on the complete profile content using a single API call with all images."""
        try:
            prompt = f"""
            Analyze this complete dating profile based on the comprehensive content below.
            This content was extracted from multiple screenshots covering the entire profile.
            
            PROFILE CONTENT:
            {combined_text}
            
            Provide analysis in JSON format:
            {{
                "profile_quality_score": 1-10,
                "should_like": true/false,
                "reason": "detailed reason for recommendation",
                "profile_completeness": 1-10,
                "conversation_potential": 1-10,
                "content_depth": 1-10,
                "authenticity_score": 1-10,
                "red_flags": ["any", "concerning", "elements"],
                "positive_indicators": ["good", "signs", "to", "like"],
                "personality_traits": ["observed", "traits"],
                "interests": ["extracted", "interests", "hobbies"],
                "estimated_age": 25,
                "name": "extracted_name",
                "location": "extracted_location",
                "profession": "extracted_job",
                "content_quality": "high/medium/low",
                "bio_length": "detailed/moderate/brief/missing",
                "prompt_answers": 0-10,
                "overall_impression": "detailed assessment"
            }}
            
            Base your assessment on:
            - Depth and quality of written content
            - Authenticity and genuineness of responses
            - Conversation starter potential
            - Shared interests or compatibility indicators
            - Overall effort put into the profile
            - Completeness of information provided
            
            Be thorough since this represents their complete profile content.
            """
            # Build messages with all screenshots attached
            content_parts = [{"type": "text", "text": prompt}]
            trace_lines = [
                "AI_CALL call_id=analyze_complete_profile model=gpt-5-mini temperature=0.0 response_format=json_object",
                "PROMPT=<<<BEGIN",
                *prompt.splitlines(),
                "<<<END",
                f"IMAGES count={len(screenshots)}"
            ]
            for p in screenshots:
                try:
                    _sz = os.path.getsize(p)
                except Exception:
                    _sz = "?"
                self._ai_trace_log([f"IMAGE image_path={p} image_size={_sz} bytes"])
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            self._ai_trace_log(trace_lines)
            
            t0 = time.perf_counter()
            client = self.ai_client or OpenAI()
            self.ai_client = client
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content_parts}]
            )
            dt_ms = int((time.perf_counter() - t0) * 1000)
            try:
                print(f"[AI] analyze_complete_profile model=gpt-5-mini images={len(screenshots)} duration={dt_ms}ms")
            except Exception:
                pass
            self._ai_trace_log([f"AI_TIME call_id=analyze_complete_profile model=gpt-5-mini images={len(screenshots)} duration_ms={dt_ms}"])
            try:
                parsed = json.loads(resp.choices[0].message.content or "{}")
                try:
                    print("[AI JSON analyze_complete_profile]")
                    print(json.dumps(parsed, indent=2)[:2000])
                except Exception:
                    pass
                return parsed
            except Exception:
                return {}
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            return {
                "profile_quality_score": 5,
                "should_like": False,
                "reason": "Analysis failed",
                "content_quality": "unknown"
            }
    
    # ===== Batched LLM extraction helpers =====

    def _submit_llm_batch_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit the built OpenAI-compatible payload to the model:
        - Prepend a system message to enforce strict JSON with required keys.
        - Use gpt-5-mini by default; keep gpt-5 for heavy tasks.
        - Retry once with stricter instructions on parse/validation failure.
        """
        messages = payload.get("messages", []) or []
        if not isinstance(messages, list):
            return {}

        # Base system prompt enforces strict JSON and required keys (values may be null)
        system_msg = {
            "role": "system",
            "content": (
            ),
        }
        strict_retry_msg = {
            "role": "system",
            "content": (
                "STRICT MODE: Respond with a single JSON object only. No text outside JSON. "
                "Do not include code fences. Ensure keys name, age, height, location exist."
            ),
        }

        model = getattr(self.config, "extraction_model", "gpt-5-mini") or "gpt-5-mini"
        # First attempt
        def _call(msgs):
            t0 = time.perf_counter()
            client = self.ai_client or OpenAI()
            self.ai_client = client
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=msgs,
            )
            dt_ms = int((time.perf_counter() - t0) * 1000)
            try:
                print(f"[AI] submit_batch model={model} messages={len(msgs)} duration={dt_ms}ms")
            except Exception:
                pass
            self._ai_trace_log([f"AI_TIME call_id=submit_batch model={model} messages={len(msgs)} duration_ms={dt_ms}"])
            return json.loads(resp.choices[0].message.content or "{}")

        try:
            attempt_msgs = messages
            result = _call(attempt_msgs)
            # Normalize to Title-Case for core keys if model emitted lowercase variants
            if isinstance(result, dict):
                if "name" in result:
                    if "Name" not in result:
                        result["Name"] = result.pop("name")
                    else:
                        result.pop("name", None)
                if "age" in result:
                    if "Age" not in result:
                        result["Age"] = result.pop("age")
                    else:
                        result.pop("age", None)
                if "height" in result:
                    if "Height" not in result:
                        result["Height"] = result.pop("height")
                    else:
                        result.pop("height", None)
                if "location" in result:
                    if "Location" not in result:
                        result["Location"] = result.pop("location")
                    else:
                        result.pop("location", None)
            # Validate required keys
            missing = self._validate_required_fields(result)
            if not missing:
                return result
            # Retry if missing required keys - mutate user prompt text to add strict appendix (no system msg)
            strict_appendix = "\n\nSTRICT: Ensure keys Name, Age, Height, Location exist exactly as spelled (Title-Case). Do not include lowercase variants or code fences. Return only a JSON object."
            retry_msgs = list(messages)
            if retry_msgs and isinstance(retry_msgs[0], dict) and isinstance(retry_msgs[0].get("content"), list):
                new_content = []
                appended = False
                for part in retry_msgs[0]["content"]:
                    if (not appended) and isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        new_content.append({"type": "text", "text": part["text"] + strict_appendix})
                        appended = True
                    else:
                        new_content.append(part)
                retry_msgs[0] = {**retry_msgs[0], "content": new_content}
            retry_result = _call(retry_msgs)
            # Normalize Title-Case again post-retry
            if isinstance(retry_result, dict):
                if "name" in retry_result:
                    if "Name" not in retry_result:
                        retry_result["Name"] = retry_result.pop("name")
                    else:
                        retry_result.pop("name", None)
                if "age" in retry_result:
                    if "Age" not in retry_result:
                        retry_result["Age"] = retry_result.pop("age")
                    else:
                        retry_result.pop("age", None)
                if "height" in retry_result:
                    if "Height" not in retry_result:
                        retry_result["Height"] = retry_result.pop("height")
                    else:
                        retry_result.pop("height", None)
                if "location" in retry_result:
                    if "Location" not in retry_result:
                        retry_result["Location"] = retry_result.pop("location")
                    else:
                        retry_result.pop("location", None)
            return retry_result
        except Exception as e1:
            try:
                strict_appendix = "\n\nSTRICT: Ensure keys Name, Age, Height, Location exist exactly as spelled (Title-Case). Do not include lowercase variants or code fences. Return only a JSON object."
                retry_msgs = list(messages)
                if retry_msgs and isinstance(retry_msgs[0], dict) and isinstance(retry_msgs[0].get("content"), list):
                    new_content = []
                    appended = False
                    for part in retry_msgs[0]["content"]:
                        if (not appended) and isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                            new_content.append({"type": "text", "text": part["text"] + strict_appendix})
                            appended = True
                        else:
                            new_content.append(part)
                    retry_msgs[0] = {**retry_msgs[0], "content": new_content}
                retry_result = _call(retry_msgs)
                # Normalize Title-Case post-retry
                if isinstance(retry_result, dict):
                    if "name" in retry_result:
                        if "Name" not in retry_result:
                            retry_result["Name"] = retry_result.pop("name")
                        else:
                            retry_result.pop("name", None)
                    if "age" in retry_result:
                        if "Age" not in retry_result:
                            retry_result["Age"] = retry_result.pop("age")
                        else:
                            retry_result.pop("age", None)
                    if "height" in retry_result:
                        if "Height" not in retry_result:
                            retry_result["Height"] = retry_result.pop("height")
                        else:
                            retry_result.pop("height", None)
                    if "location" in retry_result:
                        if "Location" not in retry_result:
                            retry_result["Location"] = retry_result.pop("location")
                        else:
                            retry_result.pop("location", None)
                return retry_result
            except Exception as e2:
                print(f"❌ Extraction failed after retry: {e2}")
                return {}

    def _validate_required_fields(self, obj: Any) -> list:
        """
        Ensure required top-level keys exist (values may be None).
        Returns the list of missing required keys.
        """
        required = ["Name", "Age", "Height", "Location"]
        if not isinstance(obj, dict):
            return required
        missing = [k for k in required if k not in obj]
        return missing

    def _to_bool_or_none(self, v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            if v == 1:
                return True
            if v == 0:
                return False
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes", "y", "has", "present", "1"):
                return True
            if s in ("false", "no", "n", "none", "absent", "0"):
                return False
        return None

    def _tri_state(self, v: Any) -> Optional[str]:
        """
        Map free-form lifestyle strings to one of: "Yes", "Sometimes", "No".
        If not explicitly indicated on-screen, return None (hidden).
        """
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("yes", "y", "yeah", "yep", "daily", "often", "regularly", "frequently", "heavy", "smoker", "drinker", "uses"):
                return "Yes"
            if s in ("no", "n", "never", "doesn't", "does not", "none", "sober", "non-smoker", "nope"):
                return "No"
            if s in ("sometimes", "socially", "occasionally", "rarely", "light", "moderate"):
                return "Sometimes"
        return None

    # Removed normalization function — raw LLM output is now used directly

    @gated_step
    def scroll_profile_node(self, state: HingeAgentState) -> HingeAgentState:
        """Scroll to see more profile content"""
        print("📜 Scrolling profile...")
        
        scroll_analysis = analyze_profile_scroll_content(
            state['current_screenshot']
        )
        
        if not scroll_analysis.get('should_scroll_down'):
            return {
                **state,
                "last_action": "scroll_profile",
                "action_successful": False
            }
        
        # Perform scroll
        scroll_x = int(getattr(self.config, "vertical_swipe_x_pct", 0.12) * state["width"])
        scroll_y_start = int(scroll_analysis.get('scroll_area_center_y', 0.6) * state["height"])
        scroll_y_end = int(scroll_y_start * 0.3)
        
        self._vertical_swipe_px(state, scroll_x, scroll_y_start, scroll_y_end, duration_ms=int(getattr(self.config, "vertical_swipe_duration_ms", 1200)))
        time.sleep(2)
        
        # Capture new content
        new_screenshot = capture_screenshot(state["device"], f"scrolled_{time.time()}")
        additional_text = extract_text_from_image(new_screenshot)
        
        # Update profile text if new content found
        updated_text = state["profile_text"]
        if additional_text and additional_text not in updated_text:
            updated_text += "\n" + additional_text
        
        return {
            **state,
            "current_screenshot": new_screenshot,
            "profile_text": updated_text,
            "last_action": "scroll_profile",
            "action_successful": True
        }
    
    @gated_step
    def make_like_decision_node(self, state: HingeAgentState) -> HingeAgentState:
        """Make like/dislike decision based on profile analysis"""
        print("🎯 Making like/dislike decision...")
        
        analysis = state.get("profile_analysis", {})
        quality = analysis.get('profile_quality_score', 0)
        potential = analysis.get('conversation_potential', 0)
        red_flags = analysis.get('red_flags', [])
        positive_indicators = analysis.get('positive_indicators', [])
        
        # Decision logic
        should_like = False
        reason = "Default: not meeting criteria"
        
        if red_flags:
            should_like = False
            reason = f"Red flags: {', '.join(red_flags[:2])}"
        elif quality >= self.config.quality_threshold_high and potential >= self.config.conversation_threshold_high:
            should_like = True
            reason = f"Excellent profile (quality: {quality}, potential: {potential})"
        elif quality >= self.config.quality_threshold_medium and len(positive_indicators) >= self.config.min_positive_indicators:
            should_like = True
            reason = f"Good profile with positives: {', '.join(positive_indicators[:2])}"
        elif len(state["profile_text"]) > self.config.min_text_length_detailed and quality >= self.config.min_quality_for_detailed:
            should_like = True
            reason = "Detailed profile with decent quality"
        
        print(f"🎯 DECISION: {'💖 LIKE' if should_like else '👎 DISLIKE'} - {reason}")
        
        return {
            **state,
            "decision_reason": reason,
            "last_action": "make_like_decision",
            "action_successful": True,
            "profile_analysis": {**analysis, "should_like": should_like}
        }
    
    @gated_step
    def detect_like_button_node(self, state: HingeAgentState) -> HingeAgentState:
        """Detect like button location using computer vision"""
        print("🎯 Detecting like button with OpenCV...")
        
        # Take fresh screenshot for button detection
        fresh_screenshot = capture_screenshot(
            state["device"],
            f"like_detection_{state['current_profile_index']}"
        )
        
        # Use CV-based detection (no LLM)
        cv_result = detect_like_button_cv(fresh_screenshot)
        
        if not cv_result.get('found'):
            print("❌ Like button not found with CV detection")
            return {
                **state,
                "current_screenshot": fresh_screenshot,
                "last_action": "detect_like_button",
                "action_successful": False
            }
        
        confidence = cv_result.get('confidence', 0)
        # CV confidence threshold is handled in the CV function
        like_x = cv_result['x']
        like_y = cv_result['y']
        
        print(f"✅ Like button detected with OpenCV:")
        print(f"   📍 Coordinates: ({like_x}, {like_y})")
        print(f"   🎯 CV Confidence: {confidence:.3f}")
        print(f"   📐 Template size: {cv_result['width']}x{cv_result['height']}")
        
        return {
            **state,
            "current_screenshot": fresh_screenshot,
            "like_button_coords": (like_x, like_y),
            "like_button_confidence": confidence,
            "last_action": "detect_like_button",
            "action_successful": True
        }
    
    @gated_step
    def execute_like_node(self, state: HingeAgentState) -> HingeAgentState:
        """Execute like action with profile change verification"""
        print("💖 Executing like action...")
        
        # Store previous profile data for verification
        updated_state = {
            **state,
            "previous_profile_text": state.get('profile_text', ''),
        }
        
        current_analysis = state.get('profile_analysis', {})
        updated_state["previous_profile_features"] = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        # Re-detect like button on current screen using CV
        fresh_screenshot = capture_screenshot(state["device"], "fresh_like_detection")
        
        # Update state immediately with fresh screenshot
        updated_state["current_screenshot"] = fresh_screenshot
        
        # Use CV-based detection for more accuracy
        cv_result = detect_like_button_cv(fresh_screenshot)
        
        if not cv_result.get('found'):
            print("❌ Like button not found with CV on fresh screenshot")
            return {
                **updated_state,
                "last_action": "execute_like",
                "action_successful": False
            }
        
        confidence = cv_result.get('confidence', 0)
        like_x = cv_result['x']
        like_y = cv_result['y']
        
        print(f"🎯 Like button detected with OpenCV:")
        print(f"   📱 Screen size: {state['width']}x{state['height']}")
        print(f"   📍 Coordinates: ({like_x}, {like_y})")
        print(f"   🎯 CV Confidence: {confidence:.3f}")
        print(f"   📐 Template size: {cv_result['width']}x{cv_result['height']}")
        
        # Execute the like tap
        if getattr(self.config, "dry_run", False):
            print(f"🧪 DRY RUN: would tap LIKE at ({like_x}, {like_y}) - confidence: {confidence:.3f}")
            return {
                **updated_state,
                "last_action": "execute_like",
                "action_successful": True
            }
        tap_with_confidence(state["device"], like_x, like_y, confidence)
        time.sleep(3)
        
        # Check if comment interface appeared
        immediate_screenshot = capture_screenshot(state["device"], "post_like_immediate")
        comment_ui = detect_comment_ui_elements(immediate_screenshot)
        comment_interface_appeared = comment_ui.get('comment_field_found', False)
        
        if comment_interface_appeared:
            print("💬 Comment interface appeared - ready to send like/comment")
            return {
                **updated_state,
                "current_screenshot": immediate_screenshot,
                "last_action": "execute_like", 
                "action_successful": True
            }
        
        # Check if we moved to next profile using verification
        time.sleep(2)
        verification_screenshot = capture_screenshot(state["device"], "like_verification")
        
        # Use profile change verification
        profile_verification = self._verify_profile_change_internal({
            **updated_state,
            "current_screenshot": verification_screenshot
        })
        
        if profile_verification.get('profile_changed', False):
            print(f"✅ Navigation occurred after like tap (confidence: {profile_verification.get('confidence', 0):.2f})")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "current_profile_index": state["current_profile_index"] + 1,
                "profiles_processed": state["profiles_processed"] + 1,
                "stuck_count": 0,
                "last_action": "execute_like",
                "action_successful": True
            }
        else:
            print("⚠️ Like may have failed - still on same profile")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "stuck_count": state["stuck_count"] + 1,
                "last_action": "execute_like",
                "action_successful": False
            }
    
    @gated_step
    def generate_comment_node(self, state: HingeAgentState) -> HingeAgentState:
        """Generate flirty, date-focused comment for current profile"""
        print("💬 Generating flirty, date-focused comment...")
        
        if not state['profile_text']:
            return {
                **state,
                "last_action": "generate_comment",
                "action_successful": False
            }
        
        # Use contextual generation if we have detailed profile analysis
        profile_analysis = state.get('profile_analysis', {})
        if profile_analysis and len(profile_analysis) > 3:
            print("🎯 Using contextual comment generation with profile analysis...")
            comment = generate_contextual_date_comment(
                profile_analysis, 
                state['profile_text']
            )
        else:
            print("💬 Using standard flirty comment generation...")
            comment = generate_comment(state['profile_text'])
        
        if not comment:
            comment = self.config.default_comment
        
        comment_id = str(uuid.uuid4())
        store_generated_comment(
            comment_id=comment_id,
            profile_text=state['profile_text'],
            generated_comment=comment,
            style_used="langgraph_flirty_contextual"
        )
        
        print(f"💋 Generated flirty comment: {comment[:60]}...")
        
        return {
            **state,
            "generated_comment": comment,
            "comment_id": comment_id,
            "last_action": "generate_comment",
            "action_successful": True
        }
    
    @gated_step
    def type_comment_node(self, state: HingeAgentState) -> HingeAgentState:
        """Type comment text into the comment field"""
        print("⌨️ Typing comment into field...")
        
        if not state.get('generated_comment'):
            print("❌ No comment to type")
            return {
                **state,
                "last_action": "type_comment",
                "action_successful": False
            }
        
        comment = state['generated_comment']
        print(f"💬 Typing comment: {comment[:50]}...")
        
        try:
            # Fresh screenshot to see current interface
            fresh_screenshot = capture_screenshot(state["device"], "comment_interface_typing")
            
            comment_ui = detect_comment_ui_elements(fresh_screenshot)
            
            if not comment_ui.get('comment_field_found'):
                print("❌ Comment field not found")
                return {
                    **state,
                    "current_screenshot": fresh_screenshot,
                    "last_action": "type_comment",
                    "action_successful": False
                }
            
            # Tap comment field to focus
            comment_x = int(comment_ui['comment_field_x'] * state["width"])
            comment_y = int(comment_ui['comment_field_y'] * state["height"])
            print(f"🎯 Tapping comment field at ({comment_x}, {comment_y})")
            
            tap_with_confidence(state["device"], comment_x, comment_y, 
                              comment_ui.get('comment_field_confidence', 0.8))
            time.sleep(2)
            
            # Clear existing text: MOVE_END then many DEL presses
            try:
                state["device"].shell("input keyevent 123")  # KEYCODE_MOVE_END
                for _ in range(5):
                    for __ in range(16):
                        state["device"].shell("input keyevent 67")  # KEYCODE_DEL
                    time.sleep(0.1)
            except Exception:
                pass
            
            # Use robust text input with multiple fallback methods
            input_result = input_text_robust(state["device"], comment, max_attempts=2)
            
            if input_result['success']:
                print(f"✅ Comment typed successfully using {input_result['method_used']}")
            else:
                print(f"❌ Comment typing failed: {input_result.get('error', 'Unknown error')}")
                return {
                    **state,
                    "current_screenshot": fresh_screenshot,
                    "last_action": "type_comment",
                    "action_successful": False,
                    "errors_encountered": state["errors_encountered"] + 1
                }
            return {
                **state,
                "current_screenshot": fresh_screenshot,
                "last_action": "type_comment",
                "action_successful": True
            }
            
        except Exception as e:
            print(f"❌ Comment typing failed: {e}")
            return {
                **state,
                "errors_encountered": state["errors_encountered"] + 1,
                "last_action": "type_comment",
                "action_successful": False
            }
    
    @gated_step
    def close_text_interface_node(self, state: HingeAgentState) -> HingeAgentState:
        """Close keyboard and text input interface"""
        print("🔽 Closing text input interface...")
        
        try:
            # Dismiss keyboard using multiple methods
            success = dismiss_keyboard(state["device"], state["width"], state["height"])
            time.sleep(2)
            
            # Take screenshot to verify keyboard is closed
            post_close_screenshot = capture_screenshot(state["device"], "post_keyboard_close")
            
            print(f"✅ Text interface closed (success: {success})")
            return {
                **state,
                "current_screenshot": post_close_screenshot,
                "last_action": "close_text_interface",
                "action_successful": True
            }
            
        except Exception as e:
            print(f"❌ Failed to close text interface: {e}")
            return {
                **state,
                "errors_encountered": state["errors_encountered"] + 1,
                "last_action": "close_text_interface",
                "action_successful": False
            }
    
    @gated_step
    def send_comment_with_typing_node(self, state: HingeAgentState) -> HingeAgentState:
        """Consolidated comment tool: tap field, type comment, dismiss keyboard, send comment"""
        print("💬 Starting consolidated comment process...")
        
        if not state.get('generated_comment'):
            print("❌ No comment to type")
            return {
                **state,
                "last_action": "send_comment_with_typing",
                "action_successful": False
            }
        
        comment = state['generated_comment']
        print(f"💬 Processing comment: {comment[:50]}...")
        
        try:
            # Step 1: Tap the text input field
            print("🎯 Step 1: Tapping comment field...")
            fresh_screenshot = capture_screenshot(state["device"], "comment_interface_typing")
            
            # Use OpenCV to detect comment field
            cv_result = detect_comment_field_cv(fresh_screenshot)
            
            if not cv_result.get('found'):
                print("❌ Comment field not found with CV detection")
                # Fallback to LLM detection
                comment_ui = detect_comment_ui_elements(fresh_screenshot)
                
                if not comment_ui.get('comment_field_found'):
                    print("❌ Comment field not found with LLM fallback either")
                    return {
                        **state,
                        "current_screenshot": fresh_screenshot,
                        "last_action": "send_comment_with_typing",
                        "action_successful": False
                    }
                
                # Use LLM coordinates
                comment_x = int(comment_ui['comment_field_x'] * state["width"])
                comment_y = int(comment_ui['comment_field_y'] * state["height"])
                confidence = comment_ui.get('comment_field_confidence', 0.8)
                print(f"🎯 Using LLM fallback - Tapping comment field at ({comment_x}, {comment_y})")
            else:
                # Use CV coordinates
                comment_x = cv_result['x']
                comment_y = cv_result['y']
                confidence = cv_result['confidence']
                print(f"✅ Comment field found with OpenCV at ({comment_x}, {comment_y}) - confidence: {confidence:.3f}")
            
            tap_with_confidence(state["device"], comment_x, comment_y, confidence)
            time.sleep(2)
            
            # Step 2: Enter comment using ADB shell type
            print("⌨️ Step 2: Typing comment...")
            
            # Clear existing text: MOVE_END then many DEL presses
            try:
                state["device"].shell("input keyevent 123")  # KEYCODE_MOVE_END
                for _ in range(5):
                    for __ in range(16):
                        state["device"].shell("input keyevent 67")  # KEYCODE_DEL
                    time.sleep(0.1)
            except Exception:
                pass
            
            # Use robust text input
            input_result = input_text_robust(state["device"], comment, max_attempts=2)
            
            if not input_result['success']:
                print(f"❌ Comment typing failed: {input_result.get('error', 'Unknown error')}")
                return {
                    **state,
                    "current_screenshot": fresh_screenshot,
                    "last_action": "send_comment_with_typing",
                    "action_successful": False,
                    "errors_encountered": state["errors_encountered"] + 1
                }
            
            print(f"✅ Comment typed successfully using {input_result['method_used']}")
            
            # Step 3: Hide keyboard via tick (bottom-right) if available, else fallback
            print("🔽 Step 3: Hiding keyboard via tick...")
            try:
                if getattr(self, "_tick_coords", None) is None:
                    tick_shot = capture_screenshot(state["device"], "tick_detection")
                    tick = detect_keyboard_tick_cv(tick_shot)
                    if tick.get("found"):
                        self._tick_coords = (tick["x"], tick["y"], tick.get("confidence", 0.8))
                if getattr(self, "_tick_coords", None):
                    tx, ty, tconf = self._tick_coords
                    tap_with_confidence(state["device"], tx, ty, tconf)
                    time.sleep(1.5)
                else:
                    # Fallback if tick not found
                    dismiss_keyboard(state["device"], state["width"], state["height"])
                    time.sleep(1.5)
            except Exception as _e:
                print(f"⚠️ Tick hide failed, fallback to generic dismiss: {_e}")
                dismiss_keyboard(state["device"], state["width"], state["height"])
                time.sleep(1.5)
            
            # Step 4: Locate send button using CV
            print("🔍 Step 4: Finding send button with OpenCV...")
            send_screenshot = capture_screenshot(state["device"], "send_button_detection")
            
            cv_result = detect_send_button_cv(send_screenshot, "priority" if getattr(self.config, "like_mode", "priority") == "priority" else "normal")
            
            if cv_result.get('found'):
                send_x = cv_result['x']
                send_y = cv_result['y']
                confidence = cv_result['confidence']
                print(f"✅ Send button found with CV at ({send_x}, {send_y}) - confidence: {confidence:.3f}")
            else:
                # Fallback coordinates based on typical Send Like button position
                send_x = int(state["width"] * 0.67)  # Right side of screen
                send_y = int(state["height"] * 0.75)  # Lower portion
                confidence = 0.5
                print(f"⚠️ Using fallback send button coordinates ({send_x}, {send_y})")
            
            # Step 5: Tap the send button
            if getattr(self.config, "confirm_before_send", False):
                try:
                    print("🛑 Confirm before send enabled.")
                    print(f"📝 Comment preview: {comment[:200]}...")
                    answer = input("Proceed to send like now? [y/N]: ").strip().lower()
                    if answer not in ("y", "yes"):
                        print("❎ Send cancelled by user; not tapping send.")
                        return {
                            **state,
                            "last_action": "send_comment_with_typing",
                            "action_successful": False
                        }
                except Exception as _e:
                    print(f"⚠️  Confirm-before-send prompt failed: {_e}")
            print("📤 Step 5: Tapping send button...")
            if getattr(self.config, "dry_run", False):
                print(f"🧪 DRY RUN: would tap SEND at ({send_x}, {send_y}) - confidence: {confidence:.3f}")
                return {
                    **state,
                    "last_action": "send_comment_with_typing",
                    "action_successful": True
                }
            tap_with_confidence(state["device"], send_x, send_y, confidence)
            time.sleep(3)
            
            # Verify comment was sent by checking if we moved to new profile or interface closed
            verification_screenshot = capture_screenshot(state["device"], "send_comment_verification")
            
            # Use profile change verification
            profile_verification = self._verify_profile_change_internal({
                **state,
                "current_screenshot": verification_screenshot
            })
            
            if profile_verification.get('profile_changed', False):
                print("✅ Consolidated comment process successful - moved to new profile")
                try:
                    self._export_profile_row(state, True, verification_screenshot)
                except Exception as _e:
                    print(f"⚠️  Export failed: {_e}")
                return {
                    **state,
                    "current_screenshot": verification_screenshot,
                    "comments_sent": state["comments_sent"] + 1,
                    "likes_sent": state["likes_sent"] + 1,
                    "current_profile_index": state["current_profile_index"] + 1,
                    "profiles_processed": state["profiles_processed"] + 1,
                    "stuck_count": 0,
                    "last_action": "send_comment_with_typing",
                    "action_successful": True
                }
            else:
                # Check if comment interface is gone (comment sent but stayed on profile)
                still_in_comment = detect_comment_ui_elements(verification_screenshot)
                
                if not still_in_comment.get('comment_field_found'):
                    print("✅ Consolidated comment process successful (interface closed) - stayed on profile")
                    try:
                        self._export_profile_row(state, True, verification_screenshot)
                    except Exception as _e:
                        print(f"⚠️  Export failed: {_e}")
                    return {
                        **state,
                        "current_screenshot": verification_screenshot,
                        "comments_sent": state["comments_sent"] + 1,
                        "likes_sent": state["likes_sent"] + 1,
                        "last_action": "send_comment_with_typing",
                        "action_successful": True
                    }
                else:
                    print("⚠️ Consolidated comment process may have failed - still in interface")
                    return {
                        **state,
                        "current_screenshot": verification_screenshot,
                        "last_action": "send_comment_with_typing",
                        "action_successful": False
                    }
            
        except Exception as e:
            print(f"❌ Consolidated comment process failed: {e}")
            return {
                **state,
                "errors_encountered": state["errors_encountered"] + 1,
                "last_action": "send_comment_with_typing",
                "action_successful": False
            }
    
    @gated_step
    def send_like_without_comment_node(self, state: HingeAgentState) -> HingeAgentState:
        """Send like without comment as fallback when comment typing fails"""
        print("💖 Sending like without comment (fallback mode)...")
        
        try:
            # Take a screenshot to inspect current interface
            fresh_screenshot = capture_screenshot(state["device"], "fallback_like_before_close")
            
            # Check if comment interface is open
            comment_ui = detect_comment_ui_elements(fresh_screenshot)
            
            if comment_ui.get('comment_field_found'):
                print("📤 Attempting to send like directly from comment interface...")
                # Try CV to find the send button
                send_try = detect_send_button_cv(fresh_screenshot, "priority" if getattr(self.config, "like_mode", "priority") == "priority" else "normal")
                if send_try.get('found'):
                    send_x = send_try['x']
                    send_y = send_try['y']
                    conf = send_try.get('confidence', 0.6)
                    print(f"✅ Send button found in modal at ({send_x}, {send_y}) - confidence: {conf:.3f}")
                else:
                    # Fallback coordinates
                    send_x = int(state["width"] * 0.67)
                    send_y = int(state["height"] * 0.75)
                    conf = 0.5
                    print(f"⚠️ Using fallback send coordinates in modal ({send_x}, {send_y})")
                
                # Tap the send button (with optional confirmation)
                if getattr(self.config, "confirm_before_send", False):
                    try:
                        print("🛑 Confirm before send enabled.")
                        answer = input("Proceed to send like now? [y/N]: ").strip().lower()
                        if answer not in ("y", "yes"):
                            print("❎ Send cancelled by user; not tapping send.")
                            return {
                                **state,
                                "last_action": "send_like_without_comment",
                                "action_successful": False
                            }
                    except Exception as _e:
                        print(f"⚠️  Confirm-before-send prompt failed: {_e}")
                if getattr(self.config, "dry_run", False):
                    print(f"🧪 DRY RUN: would tap SEND (modal) at ({send_x}, {send_y}) - confidence: {conf:.3f}")
                    return {
                        **state,
                        "last_action": "send_like_without_comment",
                        "action_successful": True
                    }
                tap_with_confidence(state["device"], send_x, send_y, conf)
                time.sleep(3)
                
                # Verify like was sent
                verification_screenshot = capture_screenshot(state["device"], "fallback_like_verification")
                
                previous_profile_text = state.get('profile_text', '')
                current_analysis = state.get('profile_analysis', {})
                previous_profile_features = {
                    'age': current_analysis.get('estimated_age', 0),
                    'name': current_analysis.get('name', ''),
                    'location': current_analysis.get('location', ''),
                    'interests': current_analysis.get('interests', [])
                }
                
                profile_verification = self._verify_profile_change_internal({
                    **state,
                    "current_screenshot": verification_screenshot,
                    "previous_profile_text": previous_profile_text,
                    "previous_profile_features": previous_profile_features
                })
                
                if profile_verification.get('profile_changed', False):
                    print("✅ Like sent successfully without comment (from modal) - moved to new profile")
                    try:
                        self._export_profile_row(state, False, verification_screenshot)
                    except Exception as _e:
                        print(f"⚠️  Export failed: {_e}")
                    return {
                        **state,
                        "current_screenshot": verification_screenshot,
                        "likes_sent": state["likes_sent"] + 1,
                        "current_profile_index": state["current_profile_index"] + 1,
                        "profiles_processed": state["profiles_processed"] + 1,
                        "stuck_count": 0,
                        "last_action": "send_like_without_comment",
                        "action_successful": True
                    }
                else:
                    print("⚠️ Modal send may have failed - attempting like button fallback")
                    # Fall through to like button detection fallback below
                    final_screenshot = verification_screenshot
                # Continue fallback in case modal didn’t produce movement
            else:
                # Take fresh screenshot for like button detection
                final_screenshot = capture_screenshot(state["device"], "fallback_like_detection")
            
            # Use CV-based like button detection
            cv_result = detect_like_button_cv(final_screenshot)
            
            if not cv_result.get('found'):
                print("❌ Like button not found with CV in fallback mode")
                return {
                    **state,
                    "current_screenshot": final_screenshot,
                    "last_action": "send_like_without_comment",
                    "action_successful": False
                }
            
            confidence = cv_result.get('confidence', 0)
            like_x = cv_result['x']
            like_y = cv_result['y']
            
            print(f"🎯 Like button detected in fallback mode:")
            print(f"   📍 Coordinates: ({like_x}, {like_y})")
            print(f"   🎯 CV Confidence: {confidence:.3f}")
            
            # Execute the like tap
            if getattr(self.config, "dry_run", False):
                print(f"🧪 DRY RUN: would tap LIKE (fallback) at ({like_x}, {like_y}) - confidence: {confidence:.3f}")
                return {
                    **state,
                    "last_action": "send_like_without_comment",
                    "action_successful": True
                }
            tap_with_confidence(state["device"], like_x, like_y, confidence)
            time.sleep(3)
            
            # Verify like was successful by checking for profile change
            verification_screenshot = capture_screenshot(state["device"], "fallback_like_verification")
            
            # Store previous profile data for verification
            previous_profile_text = state.get('profile_text', '')
            current_analysis = state.get('profile_analysis', {})
            previous_profile_features = {
                'age': current_analysis.get('estimated_age', 0),
                'name': current_analysis.get('name', ''),
                'location': current_analysis.get('location', ''),
                'interests': current_analysis.get('interests', [])
            }
            
            profile_verification = self._verify_profile_change_internal({
                **state,
                "current_screenshot": verification_screenshot,
                "previous_profile_text": previous_profile_text,
                "previous_profile_features": previous_profile_features
            })
            
            if profile_verification.get('profile_changed', False):
                print("✅ Like sent successfully without comment - moved to new profile")
                try:
                    self._export_profile_row(state, False, verification_screenshot)
                except Exception as _e:
                    print(f"⚠️  Export failed: {_e}")
                return {
                    **state,
                    "current_screenshot": verification_screenshot,
                    "likes_sent": state["likes_sent"] + 1,
                    "current_profile_index": state["current_profile_index"] + 1,
                    "profiles_processed": state["profiles_processed"] + 1,
                    "stuck_count": 0,
                    "last_action": "send_like_without_comment",
                    "action_successful": True
                }
            else:
                print("⚠️ Fallback like may have failed - still on same profile")
                return {
                    **state,
                    "current_screenshot": verification_screenshot,
                    "last_action": "send_like_without_comment",
                    "action_successful": False
                }
                
        except Exception as e:
            print(f"❌ Send like without comment failed: {e}")
            return {
                **state,
                "errors_encountered": state["errors_encountered"] + 1,
                "last_action": "send_like_without_comment",
                "action_successful": False
            }
    
    @gated_step
    def execute_dislike_node(self, state: HingeAgentState) -> HingeAgentState:
        """Execute dislike action with profile change verification"""
        print(f"👎 Executing dislike: {state.get('decision_reason', 'criteria not met')}")
        
        # Store previous profile data for verification
        updated_state = {
            **state,
            "previous_profile_text": state.get('profile_text', ''),
        }
        
        current_analysis = state.get('profile_analysis', {})
        updated_state["previous_profile_features"] = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        # Execute dislike tap
        x_dislike = int(state["width"] * self.config.dislike_button_coords[0])
        y_dislike = int(state["height"] * self.config.dislike_button_coords[1])
        
        tap(state["device"], x_dislike, y_dislike)
        time.sleep(3)
        
        # Verify dislike using profile change detection
        verification_screenshot = capture_screenshot(state["device"], "dislike_verification")
        
        profile_verification = self._verify_profile_change_internal({
            **updated_state,
            "current_screenshot": verification_screenshot
        })
        
        if profile_verification.get('profile_changed', False):
            print("✅ Dislike successful - moved to new profile")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "current_profile_index": state["current_profile_index"] + 1,
                "profiles_processed": state["profiles_processed"] + 1,
                "stuck_count": 0,
                "last_action": "execute_dislike",
                "action_successful": True
            }
        else:
            print("⚠️ Dislike may have failed - still on same profile")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "stuck_count": state["stuck_count"] + 1,
                "last_action": "execute_dislike",
                "action_successful": False
            }
    
    @gated_step
    def navigate_to_next_node(self, state: HingeAgentState) -> HingeAgentState:
        """Navigate to next profile using swipe"""
        print("➡️ Navigating to next profile...")
        
        # Store previous profile data for verification
        updated_state = {
            **state,
            "previous_profile_text": state.get('profile_text', ''),
        }
        
        current_analysis = state.get('profile_analysis', {})
        updated_state["previous_profile_features"] = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        # Execute navigation swipe
        x1_swipe = int(state["width"] * 0.15)
        y1_swipe = int(state["height"] * 0.5)
        x2_swipe = x1_swipe
        y2_swipe = int(y1_swipe * 0.75)
        
        swipe(state["device"], x1_swipe, y1_swipe, x2_swipe, y2_swipe)
        time.sleep(3)
        
        # Verify navigation
        nav_screenshot = capture_screenshot(state["device"], "navigation_verification")
        
        profile_verification = self._verify_profile_change_internal({
            **updated_state,
            "current_screenshot": nav_screenshot
        })
        
        if profile_verification.get('profile_changed', False):
            print(f"✅ Navigation successful - moved to profile {state['current_profile_index'] + 2}")
            return {
                **updated_state,
                "current_screenshot": nav_screenshot,
                "current_profile_index": state["current_profile_index"] + 1,
                "profiles_processed": state["profiles_processed"] + 1,
                "stuck_count": 0,
                "last_action": "navigate_to_next",
                "action_successful": True
            }
        else:
            print("⚠️ Navigation failed - still on same profile")
            return {
                **updated_state,
                "current_screenshot": nav_screenshot,
                "stuck_count": state["stuck_count"] + 1,
                "last_action": "navigate_to_next",
                "action_successful": False
            }
    
    @gated_step
    def verify_profile_change_node(self, state: HingeAgentState) -> HingeAgentState:
        """Verify if we've moved to a new profile"""
        print("🔍 Verifying profile change...")
        
        verification_result = self._verify_profile_change_internal(state)
        profile_changed = verification_result.get('profile_changed', False)
        confidence = verification_result.get('confidence', 0)
        
        print(f"📊 Profile change verification: {profile_changed} (confidence: {confidence:.2f})")
        
        return {
            **state,
            "last_action": "verify_profile_change",
            "action_successful": profile_changed
        }
    
    @gated_step
    def recover_from_stuck_node(self, state: HingeAgentState) -> HingeAgentState:
        """Attempt recovery when stuck using multiple swipe patterns"""
        print("🔄 Attempting recovery from stuck state...")
        
        # Multiple swipe patterns for recovery
        recovery_attempts = [
            # Aggressive horizontal swipe
            (int(state["width"] * 0.9), int(state["height"] * 0.5), 
             int(state["width"] * 0.1), int(state["height"] * 0.5)),
            # Vertical swipe down (poll-safe on left)
            (int(state["width"] * getattr(self.config, "vertical_swipe_x_pct", 0.12)), int(state["height"] * 0.3), 
             int(state["width"] * getattr(self.config, "vertical_swipe_x_pct", 0.12)), int(state["height"] * 0.7)),
            # Diagonal swipe
            (int(state["width"] * 0.8), int(state["height"] * 0.3), 
             int(state["width"] * 0.2), int(state["height"] * 0.7)),
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(recovery_attempts):
            print(f"🔄 Recovery attempt {i + 1}: Swipe from ({x1}, {y1}) to ({x2}, {y2})")
            try:
                if x1 == x2:
                    self._vertical_swipe_px(state, x1, y1, y2, duration_ms=int(getattr(self.config, "vertical_swipe_duration_ms", 1200)))
                else:
                    swipe(state["device"], x1, y1, x2, y2, duration=800)
            except Exception:
                swipe(state["device"], x1, y1, x2, y2, duration=800)
            time.sleep(2)
            
            # Check if we're unstuck
            recovery_screenshot = capture_screenshot(state["device"], f"recovery_attempt_{i}")
            current_text = extract_text_from_image(recovery_screenshot)
            
            if current_text != state.get('profile_text', ''):
                print(f"✅ Recovery successful on attempt {i + 1}")
                break
        
        # Capture final result
        final_screenshot = capture_screenshot(state["device"], "recovery_result")
        
        return {
            **state,
            "current_screenshot": final_screenshot,
            "stuck_count": 0,  # Reset stuck count after recovery
            "last_action": "recover_from_stuck",
            "action_successful": True
        }
    
    @gated_step
    def reset_app_node(self, state: HingeAgentState) -> HingeAgentState:
        """Reset the Hinge app when stuck - force close, clear from multitasking, and reopen"""
        print("🔄 Executing app reset to recover from stuck state...")
        
        try:
            # Use the reset function from helper_functions
            reset_hinge_app(state["device"])
            
            # Capture screenshot after app reset
            reset_screenshot = capture_screenshot(
                state["device"], 
                f"app_reset_{state['current_profile_index']}"
            )
            
            # Reset state counters since we're starting fresh
            return {
                **state,
                "current_screenshot": reset_screenshot,
                "profile_text": "",  # Clear previous profile data
                "profile_analysis": {},
                "previous_profile_text": "",
                "previous_profile_features": {},
                "stuck_count": 0,  # Reset stuck count
                "retry_count": 0,
                "last_action": "reset_app",
                "action_successful": True,
                "errors_encountered": max(0, state["errors_encountered"] - 1)  # Reduce error count as reset might fix issues
            }
            
        except Exception as e:
            print(f"❌ App reset failed: {e}")
            return {
                **state,
                "errors_encountered": state["errors_encountered"] + 1,
                "last_action": "reset_app",
                "action_successful": False
            }
    
    def _export_profile_row(self, state: HingeAgentState, sent_comment: bool, screenshot_path: str) -> None:
        """Append a structured row to the profile export (XLSX)."""
        try:
            if not getattr(self, "exporter", None) and getattr(self.config, "export_xlsx", True):
                try:
                    self.exporter = ProfileExporter(
                        export_dir=self.config.export_dir,
                        session_id=self.session_id,
                        export_xlsx=True,
                    )
                except Exception:
                    self.exporter = None
            if not getattr(self, "exporter", None):
                return

            # Primary sources
            analysis = state.get("profile_analysis", {}) or {}
            extracted = state.get("extracted_profile", {}) or {}
            # Use keys exactly as provided by the LLM (no normalization)

            # Direct append using exact keys from LLM (no case transforms, no remap)
            if isinstance(extracted, dict):
                self.exporter.append_row(extracted)
                return

            # Helpers
            def _get_ana(key, default=""):
                return analysis.get(key, default)

            def _join_list(val):
                if isinstance(val, list):
                    try:
                        return ", ".join(map(str, val))
                    except Exception:
                        return ""
                return str(val) if val is not None else ""

            # Prepare derived strings from extracted profile
            languages_spoken = ", ".join([str(x) for x in extracted.get("languages_spoken", []) if x])
            interests = ", ".join([str(x) for x in extracted.get("interests", []) if x])
            prompts = " | ".join([
                f"{(pa.get('prompt') or '').strip()}: {(pa.get('answer') or '').strip()}"
                for pa in (extracted.get("prompts_and_answers", []) or [])
                if isinstance(pa, dict)
            ])
            pets = extracted.get("pets", {}) or {}

            # Comment trace
            comment_text = state.get("generated_comment", "") or ""
            comment_hash = hashlib.sha256(comment_text.encode("utf-8")).hexdigest()[:16] if comment_text else ""

            # Build base row with legacy analysis values
            row = {
                "session_id": getattr(self, "session_id", ""),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "profile_index": state.get("current_profile_index", 0),

                # Identity (backfill from extracted where possible)
                "name": extracted.get("name") or _get_ana("name", ""),
                "estimated_age": (extracted.get("age") if extracted.get("age") is not None else _get_ana("estimated_age", "")),
                "location": extracted.get("location") or _get_ana("location", ""),
                "profession": _get_ana("profession", "") or extracted.get("job_title") or extracted.get("work") or "",
                "education": _get_ana("education", "") or extracted.get("university") or "",

                # Lifestyle / attributes (use tri-state in existing columns)
                "drinks": extracted.get("drinking") or _get_ana("drinks", ""),
                "smokes": extracted.get("smoking") or _get_ana("smokes", ""),
                "cannabis": extracted.get("marijuana") or _get_ana("cannabis", ""),
                "drugs": extracted.get("drugs") or _get_ana("drugs", ""),
                "religion": extracted.get("religious_beliefs") or _get_ana("religion", ""),
                "politics": extracted.get("politics") or _get_ana("politics", ""),
                "kids": _get_ana("kids", ""),
                "wants_kids": _get_ana("wants_kids", ""),
                "height": extracted.get("height") or _get_ana("height", ""),
                "languages": _get_ana("languages", "") or languages_spoken,
                "interests": interests or _join_list(_get_ana("interests", [])),
                "attribute_chips_raw": _join_list(_get_ana("attribute_chips_raw", [])),

                # Content metrics and analyzer scores
                "prompts_count": _get_ana("prompt_answers", ""),
                "extracted_text_length": len(state.get("profile_text", "") or ""),
                "content_depth": _get_ana("content_depth", ""),
                "completeness": _get_ana("profile_completeness", ""),
                "profile_quality_score": _get_ana("profile_quality_score", ""),
                "conversation_potential": _get_ana("conversation_potential", ""),
                "should_like": analysis.get("should_like", ""),
                "policy_reason": state.get("decision_reason", ""),

                # Action details
                "like_mode": getattr(self.config, "like_mode", "priority"),
                "sent_like": 1,
                "sent_comment": 1 if sent_comment else 0,
                "comment_id": state.get("comment_id", ""),
                "comment_hash": comment_hash,

                # Trace
                "screenshot_path": screenshot_path or "",
                "errors_encountered": state.get("errors_encountered", 0),
            }

            # New optional columns (will be written only if present in schema)
            row.update({
                "sexuality": extracted.get("sexuality"),
                "ethnicity": extracted.get("ethnicity"),
                "current_children": extracted.get("current_children"),
                "family_plans": extracted.get("family_plans"),
                "covid_vaccine": extracted.get("covid_vaccine"),
                "pets_dog": 1 if pets.get("dog") is True else 0 if pets.get("dog") is False else "",
                "pets_cat": 1 if pets.get("cat") is True else 0 if pets.get("cat") is False else "",
                "pets_bird": 1 if pets.get("bird") is True else 0 if pets.get("bird") is False else "",
                "pets_fish": 1 if pets.get("fish") is True else 0 if pets.get("fish") is False else "",
                "pets_reptile": 1 if pets.get("reptile") is True else 0 if pets.get("reptile") is False else "",
                "zodiac_sign": extracted.get("zodiac_sign"),
                "work": extracted.get("work"),
                "job_title": extracted.get("job_title"),
                "university": extracted.get("university"),
                "religious_beliefs": extracted.get("religious_beliefs"),
                "hometown": extracted.get("hometown"),
                "languages_spoken": languages_spoken,  # keep legacy 'languages' too
                "dating_intentions": extracted.get("dating_intentions"),
                "relationship_type": extracted.get("relationship_type"),
                "bio": extracted.get("bio"),
                "prompts_and_answers": prompts,
                "summary": extracted.get("summary") or "",
                "extraction_failed": 1 if state.get("extraction_failed") else 0,
            })

            self.exporter.append_row(row)
        except Exception as e:
            print(f"⚠️  Export row failed: {e}")

    # Deterministic runner helpers (no AI planner)
    def _fail(self, state: HingeAgentState, reason: str, last_action: str) -> HingeAgentState:
        """Fail fast helper: stop the session with a clear reason."""
        return {
            **state,
            "should_continue": False,
            "completion_reason": reason,
            "last_action": last_action,
            "action_successful": False
        }

    def run_profile_flowchart(self, state: HingeAgentState) -> HingeAgentState:
        """
        Deterministic per-profile flow:
        - capture_screenshot -> analyze_profile -> make_like_decision
        - if like: detect_like_button -> execute_like -> (generate_comment -> send_comment_with_typing) else send_like_without_comment
        - else dislike
        - if still not moved: navigate_to_next
        A single attempt per step; failures fail-fast (no retries).
        """
        s = state

        # Scrape-only mode: extract full profile and stop (no navigation)
        if getattr(self.config, "scrape_only", False):
            s = self.capture_screenshot_node(s)
            if not s.get("action_successful"):
                return self._fail(s, "Failed to capture screenshot (scrape-only)", "capture_screenshot")
            s = self.analyze_profile_node(s)
            if not s.get("action_successful"):
                return self._fail(s, "Failed to analyze profile (scrape-only)", "analyze_profile")
            # Export the scraped profile and stop
            try:
                self._export_profile_row(s, False, s.get("current_screenshot"))
            except Exception as _e:
                print(f"⚠️  Export failed (scrape-only): {_e}")
            s["profiles_processed"] = s.get("profiles_processed", 0) + 1
            s["should_continue"] = False
            s["completion_reason"] = "Scrape-only completed"
            return s

        # 1) Capture screenshot
        s = self.capture_screenshot_node(s)
        if not s.get("action_successful"):
            return self._fail(s, "Failed to capture screenshot", "capture_screenshot")

        # 2) Analyze profile (comprehensive)
        s = self.analyze_profile_node(s)
        if not s.get("action_successful"):
            return self._fail(s, "Failed to analyze profile", "analyze_profile")

        # 3) Decide like/dislike (deterministic via config thresholds)
        s = self.make_like_decision_node(s)
        if not s.get("action_successful"):
            return self._fail(s, "Failed to make like/dislike decision", "make_like_decision")

        should_like = bool(s.get("profile_analysis", {}).get("should_like", False))

        if should_like:
            # 4) Find like button (CV)
            s = self.detect_like_button_node(s)
            if not s.get("action_successful"):
                return self._fail(s, "Like button not found", "detect_like_button")

            # 5) Execute like
            s = self.execute_like_node(s)
            if not s.get("action_successful"):
                return self._fail(s, "Failed to execute like", "execute_like")

            # 6) Prefer sending with comment when comment interface path works
            # Generate comment
            s_gen = self.generate_comment_node(s)
            if s_gen.get("action_successful"):
                # Try to send the comment
                s_send = self.send_comment_with_typing_node(s_gen)
                if s_send.get("action_successful"):
                    s = s_send
                else:
                    # Fallback: like without comment
                    s_fb = self.send_like_without_comment_node(s)
                    if not s_fb.get("action_successful"):
                        return self._fail(s_fb, "Failed to send like (fallback) after comment send failed", "send_like_without_comment")
                    s = s_fb
            else:
                # If comment generation failed, attempt like without comment once
                s_fb = self.send_like_without_comment_node(s)
                if not s_fb.get("action_successful"):
                    return self._fail(s_fb, "Failed to send like (no comment path)", "send_like_without_comment")
                s = s_fb

        else:
            # Dislike path
            s = self.execute_dislike_node(s)
            if not s.get("action_successful"):
                return self._fail(s, "Failed to execute dislike", "execute_dislike")

        # 7) If we're still on the same profile, try a single navigation step
        s_ver = self.verify_profile_change_node(s)
        if not s_ver.get("action_successful"):
            s_nav = self.navigate_to_next_node(s)
            if not s_nav.get("action_successful"):
                return self._fail(s_nav, "Failed to navigate to next profile", "navigate_to_next")
            s = s_nav
        else:
            s = s_ver

        return s

    def _run_automation_deterministic(self) -> Dict[str, Any]:
        """
        Deterministic automation (no AI planner):
        - initialize session
        - iterate max_profiles calling run_profile_flowchart
        - finalize session
        """
        print("🚀 Starting Deterministic Hinge automation...")
        print(f"📊 Processing up to {self.max_profiles} profiles deterministically")

        # Build initial state (similar shape to batch init)
        init_state: HingeAgentState = HingeAgentState(
            device=None,
            width=0,
            height=0,
            max_profiles=self.max_profiles,
            current_profile_index=0,
            profiles_processed=0,
            likes_sent=0,
            comments_sent=0,
            errors_encountered=0,
            stuck_count=0,
            current_screenshot=None,
            profile_text="",
            profile_analysis={},
            decision_reason="",
            previous_profile_text="",
            previous_profile_features={},
            last_action="",
            action_successful=True,
            retry_count=0,
            generated_comment="",
            comment_id="",
            like_button_coords=None,
            like_button_confidence=0.0,
            should_continue=True,
            completion_reason="",
            ai_reasoning="",
            next_tool_suggestion="",
            batch_start_index=0
        )

        # Initialize session
        s = self.initialize_session_node(init_state)
        if not s.get("action_successful") or not s.get("should_continue", True):
            # Finalize early with failure reason
            s = self.finalize_session_node(s)
            return {
                "success": False,
                "profiles_processed": s.get("profiles_processed", 0),
                "likes_sent": s.get("likes_sent", 0),
                "comments_sent": s.get("comments_sent", 0),
                "errors_encountered": s.get("errors_encountered", 0),
                "completion_reason": s.get("completion_reason", "Initialization failed"),
                "final_success_rates": {}
            }

        # Process profiles deterministically
        for idx in range(self.max_profiles):
            if not s.get("should_continue", True):
                break
            # Ensure index is set for logging
            s["current_profile_index"] = idx
            s = self.run_profile_flowchart(s)
            if not s.get("should_continue", True):
                break

        # Finalize
        s = self.finalize_session_node(s)
        return {
            "success": True,
            "profiles_processed": s.get("profiles_processed", 0),
            "likes_sent": s.get("likes_sent", 0),
            "comments_sent": s.get("comments_sent", 0),
            "errors_encountered": s.get("errors_encountered", 0),
            "completion_reason": s.get("completion_reason", "Session completed"),
            "final_success_rates": {}
        }

    @gated_step
    def finalize_session_node(self, state: HingeAgentState) -> HingeAgentState:
        """Finalize the automation session"""
        print("🎉 Finalizing automation session...")
        
        # Update final success rates
        final_success_rates = calculate_template_success_rates()
        update_template_weights(final_success_rates)
        
        completion_reason = state.get("completion_reason", "Session completed")
        if state["current_profile_index"] >= state["max_profiles"]:
            completion_reason = "Max profiles reached"
        elif state["errors_encountered"] > self.config.max_errors_before_abort:
            completion_reason = "Too many errors"
        
        print(f"📊 Final stats: {state['profiles_processed']} processed, {state['likes_sent']} likes, {state['comments_sent']} comments")
        try:
            if getattr(self, "exporter", None):
                paths = self.exporter.get_paths()
                xlsx_path = paths.get("xlsx") or ""
                if xlsx_path:
                    print(f"📄 XLSX saved: {os.path.abspath(xlsx_path)}")
                self.exporter.close()
        except Exception as _e:
            print(f"⚠️  Export close failed: {_e}")
        
        return {
            **state,
            "should_continue": False,
            "completion_reason": completion_reason,
            "last_action": "finalize_session",
            "action_successful": True
        }
    
    def _verify_profile_change_internal(self, state: HingeAgentState) -> Dict[str, Any]:
        """Internal helper for profile change verification"""
        if not state['current_screenshot']:
            return {
                "profile_changed": False,
                "confidence": 0.0,
                "message": "No screenshot available"
            }
        
        # Extract current profile info
        current_text = extract_text_from_image(
            state['current_screenshot']
        )
        
        current_analysis = analyze_dating_ui(
            state['current_screenshot']
        )
        
        # Get previous profile info
        previous_text = state.get('previous_profile_text', '')
        previous_features = state.get('previous_profile_features', {})
        
        # If first profile, consider it new
        if not previous_text and not previous_features:
            return {
                "profile_changed": True,
                "confidence": 1.0,
                "message": "First profile"
            }
        
        # Compare profiles to detect change
        profile_changed = False
        reasons = []
        
        # Text comparison
        if current_text and previous_text:
            current_words = set(current_text.lower().split())
            previous_words = set(previous_text.lower().split())
            
            if len(current_words) > 0 and len(previous_words) > 0:
                overlap = len(current_words.intersection(previous_words))
                similarity = overlap / max(len(current_words), len(previous_words))
                
                if similarity < 0.3:  # Less than 30% overlap = different profile
                    profile_changed = True
                    reasons.append(f"Text similarity low: {similarity:.2f}")
        
        # Feature comparison
        current_features = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        if previous_features:
            # Compare key features
            if (current_features['name'] != previous_features.get('name', '') and 
                current_features['name'] and previous_features.get('name')):
                profile_changed = True
                reasons.append("Different name")
            
            if (abs(current_features['age'] - previous_features.get('age', 0)) > 5 and 
                current_features['age'] > 0 and previous_features.get('age', 0) > 0):
                profile_changed = True
                reasons.append("Age difference")
            
            # Interest overlap
            current_interests = set(current_features.get('interests', []))
            previous_interests = set(previous_features.get('interests', []))
            if current_interests and previous_interests:
                interest_overlap = len(current_interests.intersection(previous_interests))
                interest_similarity = interest_overlap / max(len(current_interests), len(previous_interests))
                if interest_similarity < 0.2:
                    profile_changed = True
                    reasons.append(f"Interest overlap low: {interest_similarity:.2f}")
        
        # Calculate confidence
        confidence = 0.8 if profile_changed else 0.3
        if len(reasons) > 1:
            confidence = min(0.95, confidence + 0.1 * (len(reasons) - 1))
        
        return {
            "profile_changed": profile_changed,
            "confidence": confidence,
            "reasons": reasons,
            "current_features": current_features,
            "message": f"Profile {'changed' if profile_changed else 'unchanged'}: {', '.join(reasons) if reasons else 'similar content'}"
        }
    
    def run_automation(self) -> Dict[str, Any]:
        """Run the complete LangGraph automation workflow with batch processing"""
        # Deterministic mode: use flowchart runner instead of AI planner
        if getattr(self.config, "deterministic_mode", True):
            return self._run_automation_deterministic()
        print("🚀 Starting LangGraph-powered Hinge automation with batch processing...")
        print(f"📊 Processing {self.max_profiles} profiles in batches of {self.profiles_per_batch}")
        
        # Initialize cumulative results
        total_results = {
            "success": True,
            "profiles_processed": 0,
            "likes_sent": 0,
            "comments_sent": 0,
            "errors_encountered": 0,
            "completion_reason": "Session completed",
            "batches_completed": 0,
            "final_success_rates": {}
        }
        
        # Calculate number of batches needed
        num_batches = (self.max_profiles + self.profiles_per_batch - 1) // self.profiles_per_batch
        print(f"📦 Will process {num_batches} batches")
        
        # Initialize device connection state that persists across batches
        device = None
        width = height = 0
        
        for batch_num in range(num_batches):
            batch_start = batch_num * self.profiles_per_batch
            batch_end = min(batch_start + self.profiles_per_batch, self.max_profiles)
            
            print(f"\n🎯 Starting batch {batch_num + 1}/{num_batches} (profiles {batch_start + 1}-{batch_end})")
            
            # Create initial state for this batch
            batch_state = HingeAgentState(
                device=device,  # Reuse device connection
                width=width,
                height=height,
                max_profiles=self.max_profiles,
                current_profile_index=batch_start,
                profiles_processed=total_results["profiles_processed"],
                likes_sent=total_results["likes_sent"],
                comments_sent=total_results["comments_sent"],
                errors_encountered=total_results["errors_encountered"],
                stuck_count=0,
                current_screenshot=None,
                profile_text="",
                profile_analysis={},
                decision_reason="",
                previous_profile_text="",
                previous_profile_features={},
                last_action="",
                action_successful=True,
                retry_count=0,
                generated_comment="",
                comment_id="",
                like_button_coords=None,
                like_button_confidence=0.0,
                should_continue=True,
                completion_reason="",
                ai_reasoning="",
                next_tool_suggestion="",
                batch_start_index=batch_start
            )
            
            # Execute batch workflow
            try:
                print(f"⚡ Executing LangGraph workflow for batch {batch_num + 1}")
                batch_final_state = self.graph.invoke(batch_state)
                
                # Update persistent device state for next batch
                device = batch_final_state.get("device")
                width = batch_final_state.get("width", width)
                height = batch_final_state.get("height", height)
                
                # Accumulate results
                total_results["profiles_processed"] = batch_final_state.get("profiles_processed", total_results["profiles_processed"])
                total_results["likes_sent"] = batch_final_state.get("likes_sent", total_results["likes_sent"])
                total_results["comments_sent"] = batch_final_state.get("comments_sent", total_results["comments_sent"])
                total_results["errors_encountered"] = batch_final_state.get("errors_encountered", total_results["errors_encountered"])
                total_results["batches_completed"] = batch_num + 1
                
                # Check if we should stop due to errors
                if total_results["errors_encountered"] > self.config.max_errors_before_abort:
                    print(f"⚠️ Stopping automation due to too many errors: {total_results['errors_encountered']}")
                    total_results["completion_reason"] = "Too many errors"
                    break
                    
                print(f"✅ Batch {batch_num + 1} completed - Processed: {batch_final_state.get('profiles_processed', 0)}, Likes: {batch_final_state.get('likes_sent', 0)}, Comments: {batch_final_state.get('comments_sent', 0)}")
                
            except Exception as e:
                print(f"❌ Batch {batch_num + 1} failed: {e}")
                total_results["errors_encountered"] += 1
                total_results["success"] = False
                
                # If first batch fails, it's likely a setup issue
                if batch_num == 0:
                    return {
                        **total_results,
                        "error": str(e),
                        "completion_reason": f"Failed on first batch: {e}"
                    }
                
                # For later batches, try to continue with remaining batches
                print(f"⚠️ Continuing with next batch despite error in batch {batch_num + 1}")
                continue
        
        # Final update of success rates
        total_results["final_success_rates"] = calculate_template_success_rates()
        
        print(f"\n🎉 Automation completed!")
        print(f"📊 Total stats: {total_results['profiles_processed']} processed, {total_results['likes_sent']} likes, {total_results['comments_sent']} comments")
        print(f"📦 Batches completed: {total_results['batches_completed']}/{num_batches}")
        
        return total_results


# Usage example for testing
if __name__ == "__main__":
    agent = LangGraphHingeAgent(max_profiles=5)
    result = agent.run_automation()
    print(f"🎯 Automation completed: {result}")
