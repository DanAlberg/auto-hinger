# app/main_agent.py

"""
Main entry point for the OpenAI-controlled Hinge automation agent.
This uses OpenAI to intelligently select and execute tools for dating app automation.
"""

import asyncio
import argparse
from typing import Dict, Any
import time
import json
from datetime import datetime, timezone
from pathlib import Path

from langgraph_hinge_agent import LangGraphHingeAgent
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

    # New flags
    config.like_mode = args.like_mode
    config.deterministic_mode = not args.ai_routing
    config.dry_run = getattr(args, "dry_run", False)
    # Temporary bypass for startup pre-check (like button visibility)
    try:
        config.precheck_strict = not getattr(args, "skip_precheck", False)
    except Exception:
        pass
    
    return config


def print_session_summary(result: Dict[str, Any]):
    """Print a summary of the automation session"""
    print("\n" + "="*60)
    print("🤖 LANGGRAPH + OPENAI HINGE AUTOMATION SUMMARY")
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
    """Main entry point for the OpenAI-controlled agent"""
    print("🤖 Starting LangGraph-Powered Hinge Automation Agent")
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
        print(f"🤖 AI Controller: OpenAI + LangGraph")
        print()
        
        # Create and run LangGraph-powered agent
        agent = LangGraphHingeAgent(
            max_profiles=config.max_profiles,
            config=config
        )
        
        # Run automation
        print("🎬 Starting LangGraph-powered automation workflow...")
        print("🧠 LangGraph + OpenAI will manage state and intelligently route actions...")

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
        print(f"Failed to run LangGraph automation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_sync()
    exit(exit_code)
