import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def analyze_traffic_light_stops(stopinfo_file=None):
    """Analyze traffic light stops from the stopinfo output file"""
    
    # If no file specified, use one in the same directory as this script
    if stopinfo_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stopinfo_file = os.path.join(script_dir, "stopinfo.xml")
    
    print(f"Analyzing traffic light stops from {stopinfo_file}")
    
    if not os.path.exists(stopinfo_file):
        print(f"Error: {stopinfo_file} not found")
        return None
    
    try:
        # Parse the XML file
        tree = ET.parse(stopinfo_file)
        root = tree.getroot()
        
        # Find stops caused by traffic lights
        stops = root.findall("stopinfo")
        
        if not stops:
            print("No stop information found in the file")
            print("The stopinfo.xml file exists but contains no <stopinfo> elements.")
            print("This usually means no vehicles stopped at traffic lights during the simulation.")
            print("You may need to run your simulation longer or check your traffic light configuration.")
            
            # Create an empty results directory anyway for testing
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            graphs_dir = os.path.join(results_dir, "graphs")
            os.makedirs(graphs_dir, exist_ok=True)
            
            return None
        
        print(f"Found {len(stops)} vehicle stops")
        
        # Filter stops by reason
        tl_stops = [stop for stop in stops if stop.get("reason") == "traci:trafficLightStop"]
        
        if not tl_stops:
            tl_stops = [stop for stop in stops if "traffic" in stop.get("reason", "").lower()]
        
        print(f"Found {len(tl_stops)} traffic light stops")
        
        if not tl_stops:
            print("No traffic light stops found. Analyzing all stops instead.")
            tl_stops = stops
        
        # Extract data from stops
        stop_data = []
        for stop in tl_stops:
            vehicle = stop.get("id", "")
            started = float(stop.get("started", 0))
            ended = float(stop.get("ended", 0)) if stop.get("ended") else None
            lane = stop.get("lane", "")
            
            # Duration is only available if the stop has ended
            duration = float(ended) - float(started) if ended else None
            
            stop_data.append({
                "vehicle": vehicle,
                "started": started,
                "ended": ended,
                "duration": duration,
                "lane": lane
            })
        
        # Sort by start time
        stop_data.sort(key=lambda x: x["started"])
        
        # Analyze stop patterns
        if stop_data:
            start_times = [stop["started"] for stop in stop_data]
            durations = [stop["duration"] for stop in stop_data if stop["duration"] is not None]
            
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            graphs_dir = os.path.join(results_dir, "graphs")
            
            # Create directories if they don't exist
            os.makedirs(graphs_dir, exist_ok=True)
            
            # Create visualizations
            plt.figure(figsize=(12, 10))
            
            # Plot 1: Stop start times
            plt.subplot(2, 1, 1)
            plt.plot(range(len(start_times)), start_times, 'go-', markersize=4)
            plt.xlabel('Stop Number')
            plt.ylabel('Simulation Time (s)')
            plt.title('Traffic Light Stop Timing')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Stop durations
            if durations:
                plt.subplot(2, 1, 2)
                plt.hist(durations, bins=15, color='blue', alpha=0.7)
                plt.xlabel('Stop Duration (s)')
                plt.ylabel('Frequency')
                plt.title('Traffic Light Stop Duration Distribution')
                plt.grid(True, alpha=0.3)
            else:
                print("No duration data available (stops may not have ended)")
            
            plt.tight_layout()
            
            # Save figure with full path
            fig_path = os.path.join(graphs_dir, 'traffic_light_stop_analysis.png')
            plt.savefig(fig_path)
            plt.close()
            
            print(f"Traffic light stop analysis saved to {fig_path}")
            
            # Analyze traffic light cycle patterns
            # Calculate time between consecutive stops on the same lane
            lanes = {}
            for stop in stop_data:
                lane = stop["lane"]
                if lane not in lanes:
                    lanes[lane] = []
                lanes[lane].append(stop["started"])
            
            # Sort times for each lane
            for lane in lanes:
                lanes[lane].sort()
            
            # Calculate intervals between stops
            intervals = []
            for lane, times in lanes.items():
                if len(times) > 1:
                    lane_intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                    intervals.extend(lane_intervals)
            
            if intervals:
                # Filter out unreasonably short or long intervals
                filtered_intervals = [i for i in intervals if 10 <= i <= 120]
                
                if filtered_intervals:
                    plt.figure(figsize=(10, 6))
                    plt.hist(filtered_intervals, bins=20, color='green', alpha=0.7)
                    plt.xlabel('Time Between Consecutive Stops on Same Lane (s)')
                    plt.ylabel('Frequency')
                    plt.title('Traffic Light Cycle Analysis')
                    plt.grid(True, alpha=0.3)
                    
                    # Save with full path
                    cycle_path = os.path.join(graphs_dir, 'traffic_light_cycle_histogram.png')
                    plt.savefig(cycle_path)
                    plt.close()
                    
                    avg_interval = np.mean(filtered_intervals)
                    print(f"Estimated traffic light cycle time: {avg_interval:.2f} seconds")
                    print(f"Traffic light cycle histogram saved to {cycle_path}")
                else:
                    print("No valid intervals found for cycle analysis")
            else:
                print("Not enough data to analyze traffic light cycles")
            
            # Summarize findings
            print("\nTraffic Light Stop Summary:")
            print(f"Total traffic light stops: {len(stop_data)}")
            if durations:
                print(f"Average stop duration: {np.mean(durations):.2f} seconds")
                print(f"Maximum stop duration: {np.max(durations):.2f} seconds")
                print(f"Minimum stop duration: {np.min(durations):.2f} seconds")
            
            # Analyze stop distribution by lane
            lane_counts = {}
            for stop in stop_data:
                lane = stop["lane"]
                if lane not in lane_counts:
                    lane_counts[lane] = 0
                lane_counts[lane] += 1
            
            if lane_counts:
                print("\nStop distribution by lane:")
                for lane, count in sorted(lane_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(stop_data)) * 100
                    print(f"{lane}: {count} stops ({percentage:.2f}%)")
            
            return stop_data
        else:
            print("No valid stop data found")
            return None
        
    except Exception as e:
        print(f"Error analyzing traffic light stops: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
if __name__ == "__main__":
    analyze_traffic_light_stops()