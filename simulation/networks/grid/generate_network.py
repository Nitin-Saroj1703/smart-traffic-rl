"""
Generate SUMO network from XML configuration files
"""
import os
import subprocess
import sys

def generate_network():
    """Generate the SUMO network using netconvert"""
    print("Generating SUMO network...")
    
    simulation_dir = os.path.dirname(os.path.abspath(__file__))
    
    netconvert_path = r"C:\Program Files (x86)\Eclipse\Sumo\bin\netconvert.exe"
    
    # Generate network from node and edge files
    cmd = [
        netconvert_path,
        "--node-files", f"{simulation_dir}/nodes.nod.xml",
        "--edge-files", f"{simulation_dir}/edges.edg.xml",
        "--output-file", f"{simulation_dir}/grid_network.net.xml",
        "--tls.guess", "true",
        "--tls.join", "true",
        "--junctions.corner-detail", "5",
        "--default.junctions.radius", "10"
    ]
    
    try:
        # Update PATH to include SUMO bin if not present
        sumo_bin = r"C:\Program Files (x86)\Eclipse\Sumo\bin"
        env = os.environ.copy()
        if sumo_bin not in env.get("PATH", ""):
            env["PATH"] = sumo_bin + os.pathsep + env.get("PATH", "")
            
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        print("Network generated successfully!")
        print(f"Network file: {simulation_dir}/grid_network.net.xml")
        
        if result.stdout:
            print("Output:", result.stdout)
            
    except subprocess.CalledProcessError as e:
        print("Failed to generate network!")
        print("Error:", e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("SUMO not found! Please install SUMO first.")
        print("   Ubuntu: sudo apt-get install sumo sumo-tools")
        print("   macOS: brew install sumo")
        print("   Windows: Download from https://sumo.dlr.de/docs/Downloads.php")
        sys.exit(1)

if __name__ == "__main__":
    generate_network()
