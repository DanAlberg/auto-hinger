# app/main_agent.py

"""
Main entry point for the AI-controlled Hinge automation agent.
This uses an LLM to intelligently select and execute tools for dating app automation.
"""

import asyncio
import argparse
import os
from typing import Dict, Any
import time
import json
from datetime import datetime, timezone
from pathlib import Path

from hinge_agent import HingeAgent
from helper_functions import ensure_adb_running
from agent_config import AgentConfig, DEFAULT_CONFIG, FAST_CONFIG


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hinge Automation Agent")
    
    parser.add_argument(
        "--profiles", "-p",
        type=int,
        default=10,
        help="Maximum number of profiles to process (default: 10)"
    )
    
    parser.add_argument(
        "--config", "-c",
        choices=["default", "fast"],
        default="default",
        help="Configuration preset to use (default: default)"
    )
    
    parser.add_argument(
        "--device-ip",
        type=str,
        default="127.0.0.1", 
        help="Device IP address (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        type=str,
        default=None,
        help="Override the default (large) model id for the active provider."
    )
    parser.add_argument(
        "--llm-small-model",
        dest="llm_small_model",
        type=str,
        default=None,
        help="Override the small/fast model id for the active provider."
    )

    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini"],
        default=None,
        help="LLM provider to use (default: env LLM_PROVIDER or openai)"
    )
    parser.add_argument(
        "--gemini-model",
        choices=[
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        default=None,
        help="Primary Gemini model id (Vertex)."
    )
    parser.add_argument(
        "--gemini-small-model",
        choices=[
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        default=None,
        help="Small/fast Gemini model id (Vertex)."
    )
    parser.add_argument(
        "--gemini-project",
        type=str,
        default=None,
        help="Vertex project id for Gemini."
    )
    parser.add_argument(
        "--gemini-location",
        type=str,
        default=None,
        help="Vertex location for Gemini."
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--manual-confirm", "--confirm-steps",
        dest="manual_confirm",
        action="store_true",
        help="Require confirmation for irreversible actions (LIKE/DISLIKE); log actions to logs/"
    )
    parser.add_argument(
        "--trace-ai",
        dest="trace_ai",
        action="store_true",
        help="Trace AI inputs (prompts + image paths) to logs/ai_trace_*.log"
    )

    parser.add_argument(
        "--like-mode",
        choices=["priority", "normal"],
        default="priority",
        help="Prefer 'send priority like' or 'send like' when both are available (default: priority)"
    )
    parser.add_argument(
        "--ai-routing",
        dest="ai_routing",
        action="store_true",
        help="Use AI to route actions instead of deterministic step runner"
    )
    
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Testing mode: do not tap LIKE or SEND. No actual likes/comments are sent."
    )
    parser.add_argument(
        "--skip-precheck", "--no-precheck",
        dest="skip_precheck",
        action="store_true",
        help="Bypass startup like-button visibility pre-check (temporary testing)"
    )
    
    return parser.parse_args()


def get_config(config_name: str, args) -> AgentConfig:
    """Get configuration based on name and override with args"""
    
    configs = {
        "default": DEFAULT_CONFIG,
        "fast": FAST_CONFIG
    }
    
    config = configs[config_name]
    
    # Override with command line arguments
    config.max_profiles = args.profiles
    config.device_ip = args.device_ip
    config.verbose_logging = args.verbose
    config.manual_confirm = args.manual_confirm
    # Enable AI trace: respect default from config, OR enable when flags are provided
    config.ai_trace = (getattr(config, "ai_trace", False) or args.trace_ai or args.manual_confirm)

    # LLM routing and Gemini settings (CLI overrides env overrides defaults)
    config.llm_provider = args.llm_provider or os.getenv("LLM_PROVIDER", getattr(config, "llm_provider", "openai"))
    config.gemini_model = args.gemini_model or os.getenv("GEMINI_MODEL", getattr(config, "gemini_model", "gemini-2.5-pro"))
    config.gemini_small_model = args.gemini_small_model or os.getenv("GEMINI_SMALL_MODEL", getattr(config, "gemini_small_model", "gemini-2.5-flash"))
    config.gemini_project_id = args.gemini_project or os.getenv("GEMINI_PROJECT_ID", getattr(config, "gemini_project_id", ""))
    config.gemini_location = args.gemini_location or os.getenv("GEMINI_LOCATION", getattr(config, "gemini_location", ""))
    try:
        env_use_vertex = os.getenv("GEMINI_USE_VERTEX")
        if env_use_vertex is not None:
            config.gemini_use_vertex = env_use_vertex.strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        pass

    # Model overrides (provider-agnostic)
    llm_model_override = getattr(args, "llm_model", None) or os.getenv("LLM_MODEL")
    llm_small_override = getattr(args, "llm_small_model", None) or os.getenv("LLM_SMALL_MODEL")

    if config.llm_provider == "gemini":
        config.extraction_model = llm_model_override or config.gemini_model
        config.extraction_small_model = llm_small_override or config.gemini_small_model
    else:
        config.extraction_model = llm_model_override or os.getenv("OPENAI_MODEL") or config.extraction_model
        config.extraction_small_model = llm_small_override or os.getenv("OPENAI_SMALL_MODEL") or config.extraction_small_model

    # New flags
    config.like_mode = args.like_mode
    config.deterministic_mode = not args.ai_routing
    config.dry_run = getattr(args, "dry_run", False)
    # Temporary bypass for startup pre-check (like button visibility)
    try:
        config.precheck_strict = not getattr(args, "skip_precheck", False)
    except Exception:
        pass

    # Propagate to env for llm_client
    try:
        os.environ["LLM_PROVIDER"] = str(config.llm_provider)
        if getattr(config, "extraction_model", ""):
            os.environ["LLM_MODEL"] = str(config.extraction_model)
        if getattr(config, "extraction_small_model", ""):
            os.environ["LLM_SMALL_MODEL"] = str(config.extraction_small_model)
        if getattr(config, "gemini_model", ""):
            os.environ["GEMINI_MODEL"] = str(config.gemini_model)
        if getattr(config, "gemini_small_model", ""):
            os.environ["GEMINI_SMALL_MODEL"] = str(config.gemini_small_model)
        if getattr(config, "gemini_project_id", ""):
            os.environ["GEMINI_PROJECT_ID"] = str(config.gemini_project_id)
        if getattr(config, "gemini_location", ""):
            os.environ["GEMINI_LOCATION"] = str(config.gemini_location)
        os.environ["GEMINI_USE_VERTEX"] = "1" if getattr(config, "gemini_use_vertex", True) else "0"
    except Exception:
        pass
    
    return config


def print_session_summary(result: Dict[str, Any]):
    """Print a summary of the automation session"""
    print("\n" + "="*60)
    print("🤖 HINGE AUTOMATION SUMMARY")
    print("="*60)
    print(f"📊 Profiles Processed: {result.get('profiles_processed', 0)}")
    print(f"💖 Likes Sent: {result.get('likes_sent', 0)}")
    print(f"💬 Comments Sent: {result.get('comments_sent', 0)}")
    print(f"❌ Errors Encountered: {result.get('errors_encountered', 0)}")
    print(f"🏁 Completion Reason: {result.get('completion_reason', 'Unknown')}")
    
    if result.get('final_success_rates'):
        print(f"📈 Final Success Rates: {result['final_success_rates']}")
    
    success = result.get('success', False)
    if success:
        print("✅ Session: Completed Successfully")
    else:
        print("⚠️  Session: Completed with Issues")
    
    print("="*60)


async def main():
    """Main entry point for the AI-controlled agent"""
    print("🤖 Starting Hinge Automation Agent")
    print("="*55)
    
    try:
        # Parse arguments
        args = parse_arguments()

        # Ensure ADB is running before proceeding
        ensure_adb_running()
        
        # Get configuration  
        config = get_config(args.config, args)
        
        print(f"📋 Configuration: {args.config}")
        print(f"📱 Device IP: {config.device_ip}")
        print(f"🎯 Max Profiles: {config.max_profiles}")
        print(f"🔊 Verbose Logging: {config.verbose_logging}")
        print(f" Manual Confirm Mode: {config.manual_confirm}")
        if config.manual_confirm:
            print("Manual confirmation mode ENABLED: irreversible taps (LIKE/DISLIKE) require 'y' to proceed; actions are logged.")
        print(f"📝 AI Trace: {config.ai_trace}")
        print(f"🏷️ Like Mode: {config.like_mode}")
        print(f"🧭 Deterministic Mode: {config.deterministic_mode}")
        print(f"🧪 Dry Run Mode: {getattr(config, 'dry_run', False)}")
        print(f"️ Precheck Strict: {getattr(config, 'precheck_strict', True)}")
        llm_provider = getattr(config, "llm_provider", "openai")
        if llm_provider == "gemini":
            print("AI Controller: Gemini (Vertex)")
            print(f"Gemini Model: {getattr(config, 'gemini_model', '')}")
            print(f"Gemini Small Model: {getattr(config, 'gemini_small_model', '')}")
            print(f"Gemini Project: {getattr(config, 'gemini_project_id', '') or '(env)'}")
            print(f"Gemini Location: {getattr(config, 'gemini_location', '') or '(env)'}")
        else:
            print("AI Controller: OpenAI")
        print()
        
        # Create and run Hinge agent
        agent = HingeAgent(
            max_profiles=config.max_profiles,
            config=config
        )
        
        # Run automation
        print("🎬 Starting automation workflow...")
        if llm_provider == "gemini":
            print("Gemini will manage state and intelligently route actions...")
        else:
            print("OpenAI will manage state and intelligently route actions...")

        # Timing start
        app_dir = Path(__file__).parent
        images_dir = app_dir / "images"
        logs_dir = app_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        started_at_dt = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        result = agent.run_automation()

        # Timing end and summary
        ended_at_dt = datetime.now(timezone.utc)
        duration_seconds = round(time.perf_counter() - start_perf, 3)
        mode = "dry_run" if getattr(config, "dry_run", False) else "default"
        summary = {
            "started_at": started_at_dt.isoformat(),
            "ended_at": ended_at_dt.isoformat(),
            "duration_seconds": duration_seconds,
            "mode": mode,
            "profiles_requested": config.max_profiles,
            "profiles_processed": result.get("profiles_processed") if isinstance(result, dict) else None,
            "images_dir": images_dir.as_posix(),
            "logs_dir": logs_dir.as_posix(),
            "exit_status": "success"
        }
        ts = started_at_dt.strftime("%Y%m%d_%H%M%S")
        summary_path = logs_dir / f"run_summary_{ts}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Human-readable and machine-readable summaries
        print(f'RUN SUMMARY: duration={duration_seconds}s, mode={mode}, profiles={config.max_profiles}, images={images_dir.as_posix()}, logs={logs_dir.as_posix()}')
        print("[RUN SUMMARY] " + json.dumps(summary, ensure_ascii=False))
        
        # Print summary
        print_session_summary(result)
        
        # Return success
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Automation interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Automation failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        
        return 1


def run_sync():
    """Synchronous wrapper for the main function"""
    try:
        return asyncio.run(main())
    except Exception as e:
        print(f"Failed to run automation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_sync()
    exit(exit_code)
