#!/usr/bin/env python3
"""
Wrapper script to run MuJoCo simulations with software rendering
"""

import os
import sys
import subprocess
import argparse

def setup_software_rendering_env():
    """Set up environment variables for software rendering"""
    env = os.environ.copy()
    
    # Try different backends - start with glfw which might work better
    env['MUJOCO_GL'] = 'glfw'
    env['LIBGL_ALWAYS_SOFTWARE'] = '1'
    env['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    
    # Additional variables that might help
    env['DISPLAY'] = ':99'  # Virtual display
    env['MESA_GLSL_CACHE_DISABLE'] = '1'
    env['MESA_DEBUG'] = '1'  # Enable debug output
    env['LIBGL_ALWAYS_SOFTWARE'] = '1'
    env['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    env['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    
    return env

def run_with_xvfb(script_path, *args):
    """Run a Python script with xvfb for virtual display"""
    cmd = ['xvfb-run', '-a', 'python', script_path] + list(args)
    
    print(f"Running: {' '.join(cmd)}")
    print("Environment variables:")
    env = setup_software_rendering_env()
    for key, value in env.items():
        if key in ['MUJOCO_GL', 'LIBGL_ALWAYS_SOFTWARE', 'MESA_GL_VERSION_OVERRIDE', 'DISPLAY']:
            print(f"  {key}={value}")
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def run_direct(script_path, *args):
    """Run a Python script directly with software rendering environment"""
    cmd = ['python', script_path] + list(args)
    
    print(f"Running: {' '.join(cmd)}")
    print("Environment variables:")
    env = setup_software_rendering_env()
    for key, value in env.items():
        if key in ['MUJOCO_GL', 'LIBGL_ALWAYS_SOFTWARE', 'MESA_GL_VERSION_OVERRIDE', 'DISPLAY']:
            print(f"  {key}={value}")
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run MuJoCo simulation with software rendering')
    parser.add_argument('script', help='Python script to run')
    parser.add_argument('--no-xvfb', action='store_true', help='Run without xvfb (use direct rendering)')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Additional arguments for the script')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.script):
        print(f"Error: Script {args.script} not found")
        sys.exit(1)
    
    print("MuJoCo Software Rendering Wrapper")
    print("="*50)
    
    success = False
    if args.no_xvfb:
        print("Running without xvfb (direct rendering)...")
        success = run_direct(args.script, *args.args)
    else:
        print("Running with xvfb (virtual display)...")
        success = run_with_xvfb(args.script, *args.args)
    
    if success:
        print("\n✅ Script completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Script failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 