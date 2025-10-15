# app/langgraph_hinge_agent.py

"""
LangGraph-powered Hinge automation agent that replaces GeminiAgentController.
Uses state-based workflow management for improved reliability and debugging.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
import base64
from openai import OpenAI
import os
from datetime import datetime
from functools import wraps

from helper_functions import (
    connect_device, get_screen_resolution, open_hinge, reset_hinge_app,
    capture_screenshot, tap, tap_with_confidence, swipe,
    dismiss_keyboard, clear_screenshots_directory, detect_like_button_cv, detect_send_button_cv, detect_comment_field_cv, input_text_robust
)
from analyzer import (
    extract_text_from_image, analyze_dating_ui,
    find_ui_elements, analyze_profile_scroll_content,
    detect_comment_ui_elements, generate_comment, generate_contextual_date_comment
)
from data_store import store_generated_comment, calculate_template_success_rates
from prompt_engine import update_template_weights
from profile_export import ProfileExporter
import hashlib


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
    LangGraph-powered Hinge automation agent with Gemini-controlled decision making.
    Replaces GeminiAgentController with improved workflow management.
    """
    
    def __init__(self, max_profiles: int = 10, config=None):
        from agent_config import DEFAULT_CONFIG
        
        self.max_profiles = max_profiles
        self.config = config or DEFAULT_CONFIG
        self.ai_client = OpenAI()
        self.graph = self._build_workflow()

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
        try:
            self.exporter = ProfileExporter(
                export_dir=self.config.export_dir,
                session_id=self.session_id,
                export_csv=getattr(self.config, "export_csv", True),
                export_xlsx=getattr(self.config, "export_xlsx", False),
            )
        except Exception:
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
        """Build the LangGraph workflow with Gemini-controlled decision making"""
        
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
        
        # Add edges back to Gemini decision node from all action nodes
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
        
        # Clear old screenshots to prevent confusion
        clear_screenshots_directory()
        
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
                    "AI_CALL call_id=ai_decide_action model=gpt-4o-mini temperature=0.0 response_format=json_object",
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
                    "AI_CALL call_id=ai_decide_action model=gpt-4o-mini temperature=0.0 response_format=json_object",
                    "PROMPT=<<<BEGIN",
                    *prompt.splitlines(),
                    "<<<END",
                ])
            
            resp = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.0
            )
            
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
        """Comprehensive profile analysis with multiple scrolls to capture all content"""
        print("🔍 Starting comprehensive profile analysis...")
        
        if not state['current_screenshot']:
            return {
                **state,
                "last_action": "analyze_profile",
                "action_successful": False
            }
        
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
                "AI_CALL call_id=extract_user_content_only model=gpt-4o-mini temperature=0.0",
                "PROMPT=<<<BEGIN",
                *prompt.splitlines(),
                "<<<END",
                f"IMAGE image_path={screenshot_path} image_size={_sz} bytes"
            ])
            resp = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                temperature=0.0
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
    
    def _analyze_complete_profile(self, screenshots: list, combined_text: str) -> dict:
        """Perform comprehensive analysis on the complete profile content"""
        try:
            # Use the most recent screenshot for visual analysis
            with open(screenshots[-1], 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            
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
            
            try:
                _sz = os.path.getsize(screenshots[-1])
            except Exception:
                _sz = "?"
            self._ai_trace_log([
                "AI_CALL call_id=analyze_complete_profile model=gpt-4o-mini temperature=0.0 response_format=json_object",
                "PROMPT=<<<BEGIN",
                *prompt.splitlines(),
                "<<<END",
                f"IMAGE image_path={screenshots[-1]} image_size={_sz} bytes"
            ])
            resp = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                temperature=0.0
            )
            
            try:
                return json.loads(resp.choices[0].message.content or "{}")
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
        scroll_x = int(scroll_analysis.get('scroll_area_center_x', 0.5) * state["width"])
        scroll_y_start = int(scroll_analysis.get('scroll_area_center_y', 0.6) * state["height"])
        scroll_y_end = int(scroll_y_start * 0.3)
        
        swipe(state["device"], scroll_x, scroll_y_start, scroll_x, scroll_y_end)
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
        
        # Use CV-based detection instead of Gemini
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
            
            # Clear any existing text
            state["device"].shell("input keyevent KEYCODE_CTRL_A")
            time.sleep(0.5)
            
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
                # Fallback to Gemini detection
                comment_ui = detect_comment_ui_elements(fresh_screenshot)
                
                if not comment_ui.get('comment_field_found'):
                    print("❌ Comment field not found with Gemini fallback either")
                    return {
                        **state,
                        "current_screenshot": fresh_screenshot,
                        "last_action": "send_comment_with_typing",
                        "action_successful": False
                    }
                
                # Use Gemini coordinates
                comment_x = int(comment_ui['comment_field_x'] * state["width"])
                comment_y = int(comment_ui['comment_field_y'] * state["height"])
                confidence = comment_ui.get('comment_field_confidence', 0.8)
                print(f"🎯 Using Gemini fallback - Tapping comment field at ({comment_x}, {comment_y})")
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
            
            # Clear any existing text
            state["device"].shell("input keyevent KEYCODE_CTRL_A")
            time.sleep(0.5)
            
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
            
            # Step 3: Exit text input by tapping outside keyboard
            print("🔽 Step 3: Dismissing keyboard...")
            
            dismiss_keyboard(state["device"], state["width"], state["height"])
            time.sleep(2)
            
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
                
                # Tap the send button
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
            # Vertical swipe down
            (int(state["width"] * 0.5), int(state["height"] * 0.3), 
             int(state["width"] * 0.5), int(state["height"] * 0.7)),
            # Diagonal swipe
            (int(state["width"] * 0.8), int(state["height"] * 0.3), 
             int(state["width"] * 0.2), int(state["height"] * 0.7)),
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(recovery_attempts):
            print(f"🔄 Recovery attempt {i + 1}: Swipe from ({x1}, {y1}) to ({x2}, {y2})")
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
        """Append a structured row to the profile export (CSV/XLSX)."""
        try:
            if not getattr(self, "exporter", None):
                return
            analysis = state.get("profile_analysis", {}) or {}
            def _get(key, default=""):
                return analysis.get(key, default)
            def _join_list(val):
                if isinstance(val, list):
                    try:
                        return ", ".join(map(str, val))
                    except Exception:
                        return ""
                return str(val) if val is not None else ""
            comment_text = state.get("generated_comment", "") or ""
            comment_hash = hashlib.sha256(comment_text.encode("utf-8")).hexdigest()[:16] if comment_text else ""
            row = {
                "session_id": getattr(self, "session_id", ""),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "profile_index": state.get("current_profile_index", 0),
                "name": _get("name", ""),
                "estimated_age": _get("estimated_age", ""),
                "location": _get("location", ""),
                "profession": _get("profession", ""),
                "education": _get("education", ""),
                "drinks": _get("drinks", ""),
                "smokes": _get("smokes", ""),
                "cannabis": _get("cannabis", ""),
                "drugs": _get("drugs", ""),
                "religion": _get("religion", ""),
                "politics": _get("politics", ""),
                "kids": _get("kids", ""),
                "wants_kids": _get("wants_kids", ""),
                "height": _get("height", ""),
                "languages": _join_list(_get("languages", [])),
                "interests": _join_list(_get("interests", [])),
                "attribute_chips_raw": _join_list(_get("attribute_chips_raw", [])),
                "prompts_count": _get("prompt_answers", ""),
                "extracted_text_length": len(state.get("profile_text", "") or ""),
                "content_depth": _get("content_depth", ""),
                "completeness": _get("profile_completeness", ""),
                "profile_quality_score": _get("profile_quality_score", ""),
                "conversation_potential": _get("conversation_potential", ""),
                "should_like": analysis.get("should_like", ""),
                "policy_reason": state.get("decision_reason", ""),
                "like_mode": getattr(self.config, "like_mode", "priority"),
                "sent_like": 1,
                "sent_comment": 1 if sent_comment else 0,
                "comment_id": state.get("comment_id", ""),
                "comment_hash": comment_hash,
                "screenshot_path": screenshot_path or "",
                "errors_encountered": state.get("errors_encountered", 0),
            }
            self.exporter.append_row(row)
        except Exception as e:
            print(f"⚠️  Export row failed: {e}")

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
                csv_path = paths.get("csv") or ""
                xlsx_path = paths.get("xlsx") or ""
                if csv_path:
                    print(f"📄 CSV saved: {os.path.abspath(csv_path)}")
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
