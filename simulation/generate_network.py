"""
Create SUMO network and traffic demand files
"""
import os
import sys
import subprocess

def create_simple_network(output_dir: str = "simulation/networks/simple"):
    """
    Create a simple 4-way intersection network
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create node file
    node_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="center" x="0.0" y="0.0" type="traffic_light"/>
    <node id="north" x="0.0" y="100.0" type="priority"/>
    <node id="south" x="0.0" y="-100.0" type="priority"/>
    <node id="east" x="100.0" y="0.0" type="priority"/>
    <node id="west" x="-100.0" y="0.0" type="priority"/>
</nodes>"""
    
    with open(f"{output_dir}/nodes.nod.xml", "w") as f:
        f.write(node_content)
    
    # Create edge file
    edge_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge from="north" to="center" id="north_in" numLanes="2" speed="13.89"/>
    <edge from="center" to="north" id="north_out" numLanes="2" speed="13.89"/>
    <edge from="south" to="center" id="south_in" numLanes="2" speed="13.89"/>
    <edge from="center" to="south" id="south_out" numLanes="2" speed="13.89"/>
    <edge from="east" to="center" id="east_in" numLanes="2" speed="13.89"/>
    <edge from="center" to="east" id="east_out" numLanes="2" speed="13.89"/>
    <edge from="west" to="center" id="west_in" numLanes="2" speed="13.89"/>
    <edge from="center" to="west" id="west_out" numLanes="2" speed="13.89"/>
</edges>"""
    
    with open(f"{output_dir}/edges.edg.xml", "w") as f:
        f.write(edge_content)
    
    # Generate network with netconvert
    subprocess.run([
        "netconvert",
        "--node-files", f"{output_dir}/nodes.nod.xml",
        "--edge-files", f"{output_dir}/edges.edg.xml",
        "--output-file", f"{output_dir}/network.net.xml"
    ])
    
    print(f"Network created at {output_dir}/network.net.xml")

def create_traffic_demand(network_file: str, output_dir: str):
    """
    Create traffic demand files
    """
    # Create route file
    route_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89" color="1,0,0"/>
    
    <!-- North to South -->
    <route id="ns" edges="north_in south_out"/>
    <!-- South to North -->
    <route id="sn" edges="south_in north_out"/>
    <!-- East to West -->
    <route id="ew" edges="east_in west_out"/>
    <!-- West to East -->
    <route id="we" edges="west_in east_out"/>
    
    <!-- Traffic flows -->
    <flow id="flow_ns" route="ns" begin="0" end="3600" vehsPerHour="500" departSpeed="10" departLane="0"/>
    <flow id="flow_sn" route="sn" begin="0" end="3600" vehsPerHour="400" departSpeed="10" departLane="0"/>
    <flow id="flow_ew" route="ew" begin="0" end="3600" vehsPerHour="300" departSpeed="10" departLane="0"/>
    <flow id="flow_we" route="we" begin="0" end="3600" vehsPerHour="350" departSpeed="10" departLane="0"/>
</routes>"""
    
    with open(f"{output_dir}/routes.rou.xml", "w") as f:
        f.write(route_content)
    
    # Create SUMO configuration file
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>"""
    
    with open(f"{output_dir}/sumo_config.sumocfg", "w") as f:
        f.write(config_content)
    
    print(f"Traffic demand created at {output_dir}/")

if __name__ == "__main__":
    network_dir = "simulation/networks/simple"
    create_simple_network(network_dir)
    create_traffic_demand(f"{network_dir}/network.net.xml", network_dir)
