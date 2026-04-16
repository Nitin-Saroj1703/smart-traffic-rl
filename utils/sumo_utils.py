"""
Utility functions for interacting with SUMO simulation
"""
import traci
import sumolib
from typing import List, Dict, Optional, Tuple
import numpy as np

class SUMOUtils:
    """Helper class for SUMO simulation interactions"""
    
    def __init__(self):
        self.emergency_vehicle_ids = set()
        self.tracked_lanes = {}
        
    def get_queue_length(self, lane_id: str) -> int:
        """
        Get the number of vehicles waiting in a lane
        Waiting vehicles are those with speed < 0.1 m/s
        
        Args:
            lane_id: SUMO lane identifier
            
        Returns:
            Number of waiting vehicles
        """
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            queue_count = 0
            for veh_id in vehicles:
                if traci.vehicle.getSpeed(veh_id) < 0.1:
                    queue_count += 1
            return queue_count
        except traci.exceptions.TraCIException:
            return 0
    
    def get_avg_speed(self, lane_id: str) -> float:
        """
        Get average speed of vehicles in a lane
        
        Args:
            lane_id: SUMO lane identifier
            
        Returns:
            Average speed in m/s
        """
        try:
            return traci.lane.getLastStepMeanSpeed(lane_id)
        except traci.exceptions.TraCIException:
            return 0.0
    
    def get_co2_emission(self, lane_id: str) -> float:
        """
        Get CO2 emissions for a lane in mg/s
        
        Args:
            lane_id: SUMO lane identifier
            
        Returns:
            CO2 emissions in mg/s
        """
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            total_co2 = 0.0
            for veh_id in vehicles:
                total_co2 += traci.vehicle.getCO2Emission(veh_id)
            return total_co2
        except traci.exceptions.TraCIException:
            return 0.0
    
    def get_waiting_time(self, lane_id: str) -> float:
        """
        Get total waiting time for vehicles in a lane
        
        Args:
            lane_id: SUMO lane identifier
            
        Returns:
            Total waiting time in seconds
        """
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            total_wait = 0.0
            for veh_id in vehicles:
                total_wait += traci.vehicle.getWaitingTime(veh_id)
            return total_wait
        except traci.exceptions.TraCIException:
            return 0.0
    
    def detect_emergency_vehicle(self, lane_ids: List[str]) -> bool:
        """
        Check if any emergency vehicle is present in given lanes
        
        Args:
            lane_ids: List of lane IDs to check
            
        Returns:
            True if emergency vehicle detected, False otherwise
        """
        try:
            for lane_id in lane_ids:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                for veh_id in vehicles:
                    vehicle_type = traci.vehicle.getTypeID(veh_id)
                    if vehicle_type == "emergency":
                        return True
            return False
        except traci.exceptions.TraCIException:
            return False
    
    def get_all_intersections(self) -> List[str]:
        """
        Get list of all traffic light intersections in the network
        
        Returns:
            List of traffic light IDs
        """
        try:
            return traci.trafficlight.getIDList()
        except traci.exceptions.TraCIException:
            return []
    
    def get_incoming_lanes(self, intersection_id: str) -> List[str]:
        """
        Get incoming lanes for a specific intersection
        
        Args:
            intersection_id: Traffic light ID
            
        Returns:
            List of incoming lane IDs
        """
        try:
            return traci.trafficlight.getControlledLanes(intersection_id)
        except traci.exceptions.TraCIException:
            return []
    
    def get_traffic_light_state(self, tl_id: str) -> str:
        """
        Get current state of a traffic light
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            Current phase state string
        """
        try:
            return traci.trafficlight.getRedYellowGreenState(tl_id)
        except traci.exceptions.TraCIException:
            return ""
    
    def set_traffic_light_phase(self, tl_id: str, phase_index: int) -> None:
        """
        Set traffic light to specific phase
        
        Args:
            tl_id: Traffic light ID
            phase_index: Index of the phase to set
        """
        try:
            traci.trafficlight.setPhase(tl_id, phase_index)
        except traci.exceptions.TraCIException:
            pass
    
    def get_vehicle_count(self, lane_id: str) -> int:
        """
        Get total number of vehicles in a lane
        
        Args:
            lane_id: SUMO lane identifier
            
        Returns:
            Number of vehicles
        """
        try:
            return traci.lane.getLastStepVehicleNumber(lane_id)
        except traci.exceptions.TraCIException:
            return 0
    
    def get_lane_length(self, lane_id: str) -> float:
        """
        Get length of a lane
        
        Args:
            lane_id: SUMO lane identifier
            
        Returns:
            Lane length in meters
        """
        try:
            return traci.lane.getLength(lane_id)
        except traci.exceptions.TraCIException:
            return 0.0
    
    def get_intersection_metrics(self, intersection_id: str) -> Dict:
        """
        Get comprehensive metrics for an intersection
        
        Args:
            intersection_id: Traffic light ID
            
        Returns:
            Dictionary with various metrics
        """
        lanes = self.get_incoming_lanes(intersection_id)
        
        metrics = {
            'intersection_id': intersection_id,
            'total_queue': 0,
            'total_waiting_time': 0.0,
            'total_co2': 0.0,
            'avg_speed': 0.0,
            'vehicle_count': 0
        }
        
        for lane in lanes:
            metrics['total_queue'] += self.get_queue_length(lane)
            metrics['total_waiting_time'] += self.get_waiting_time(lane)
            metrics['total_co2'] += self.get_co2_emission(lane)
            metrics['avg_speed'] += self.get_avg_speed(lane)
            metrics['vehicle_count'] += self.get_vehicle_count(lane)
            
        if len(lanes) > 0:
            metrics['avg_speed'] /= len(lanes)
            
        return metrics
