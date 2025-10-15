# app/main_agent.py

"""
Main entry point for the OpenAI-controlled Hinge automation agent.
This uses OpenAI to intelligently select and execute tools for dating app automation.
"""

import asyncio
import argparse
from typing import Dict, Any

from langgraph_hinge_agent import LangGraphHingeAgent
from agent_config import AgentConfig, DEFAULT_CONFIG, FAST_CONFIG, CONSERVATIVE_CONFIG


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
        choices=["default", "fast", "conservative"],
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
        "--no-screenshots",
        action="store_true",
        help="Disable screenshot saving"
    )
    parser.add_argument(
        "--manual-confirm", "--confirm-steps",
        dest="manual_confirm",
        action="store_true",
        help="Require manual confirmation before each step and log all actions"
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
        "--export-xlsx",
        dest="export_xlsx",
        action="store_true",
        help="Also export an Excel file at the end (requires pandas/openpyxl)"
    )
    
    return parser.parse_args()


def get_config(config_name: str, args) -> AgentConfig:
    """Get configuration based on name and override with args"""
    
    configs = {
        "default": DEFAULT_CONFIG,
        "fast": FAST_CONFIG,
        "conservative": CONSERVATIVE_CONFIG
    }
    
    config = configs[config_name]
    
    # Override with command line arguments
    config.max_profiles = args.profiles
    config.device_ip = args.device_ip
    config.verbose_logging = args.verbose
    config.save_screenshots = not args.no_screenshots
    config.manual_confirm = args.manual_confirm
    # Enable AI trace if explicitly requested or when manual confirm mode is on
    config.ai_trace = args.trace_ai or args.manual_confirm

    # New flags
    config.like_mode = args.like_mode
    config.deterministic_mode = not args.ai_routing
    config.export_xlsx = args.export_xlsx
    
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
        
        # Get configuration  
        config = get_config(args.config, args)
        
        print(f"📋 Configuration: {args.config}")
        print(f"📱 Device IP: {config.device_ip}")
        print(f"🎯 Max Profiles: {config.max_profiles}")
        print(f"🔊 Verbose Logging: {config.verbose_logging}")
        print(f"📸 Save Screenshots: {config.save_screenshots}")
        print(f"🛑 Manual Confirm Mode: {config.manual_confirm}")
        if config.manual_confirm:
            print("Manual confirmation mode ENABLED: each step requires 'y' to proceed and all actions are logged.")
        print(f"📝 AI Trace: {config.ai_trace}")
        print(f"🏷️ Like Mode: {config.like_mode}")
        print(f"🧭 Deterministic Mode: {config.deterministic_mode}")
        print(f"📊 Export XLSX: {config.export_xlsx}")
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
        result = agent.run_automation()
        
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
