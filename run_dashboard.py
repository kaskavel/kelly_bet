#!/usr/bin/env python3
"""
Launch script for the Kelly Trading Dashboard
Usage: python run_dashboard.py
"""

import subprocess
import sys
import os
import time


def setup_streamlit_config():
    """Create Streamlit config to skip email prompt"""
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    if not os.path.exists(config_file):
        config_content = """[server]
port = 8501
address = "localhost"
headless = true

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#1f77b4"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
"""
        with open(config_file, 'w') as f:
            f.write(config_content)


def main():
    """Launch the Streamlit dashboard"""
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Setup Streamlit configuration
    setup_streamlit_config()
    
    # Set PYTHONPATH to include src directory
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(project_dir, 'src')
    
    # Simple command without shell complexities
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/dashboard.py"
    ]
    
    try:
        print("Starting Kelly Trading Dashboard...")
        print("Dashboard will be available at: http://localhost:8501")
        print("Auto-refresh enabled with 15-minute intervals")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the process
        process = subprocess.Popen(
            cmd, 
            env=env, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send empty email to skip setup
        try:
            process.stdin.write("\n")
            process.stdin.flush()
        except:
            pass
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("\nTry running directly with:")
        print("streamlit run src/ui/dashboard.py")
        sys.exit(1)


if __name__ == "__main__":
    main()