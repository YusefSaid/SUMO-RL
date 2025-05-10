#!/usr/bin/env python3
"""
Generate a simple but reliable route file for SUMO simulation
that ensures vehicles will encounter traffic lights.
"""
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def find_valid_routes(net_file, num_routes=5):
    """Find valid connected routes through traffic light junctions"""
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Find traffic light junctions
        tl_junctions = []
        for junction in root.findall('junction'):
            if junction.get('type') == 'traffic_light':
                tl_junctions.append(junction.get('id'))
        
        print(f"Found {len(tl_junctions)} traffic light junctions: {', '.join(tl_junctions)}")
        
        # Find all edges
        edges = {}
        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            # Skip internal edges
            if edge_id and not edge_id.startswith(':'):
                from_junction = edge.get('from')
                to_junction = edge.get('to')
                if from_junction and to_junction:
                    edges[edge_id] = {'from': from_junction, 'to': to_junction}
        
        # Find edge connections
        connections = {}
        for connection in root.findall('connection'):
            from_edge = connection.get('from')
            to_edge = connection.get('to')
            if from_edge and to_edge and not from_edge.startswith(':') and not to_edge.startswith(':'):
                if from_edge not in connections:
                    connections[from_edge] = []
                connections[from_edge].append(to_edge)
        
        # Create valid routes passing through traffic light junctions
        valid_routes = []
        
        # Look for edges connected to traffic light junctions
        for junction_id in tl_junctions:
            # Find incoming edges to this junction
            incoming_edges = [edge_id for edge_id, data in edges.items() 
                             if data.get('to') == junction_id]
            
            # Find outgoing edges from this junction
            outgoing_edges = [edge_id for edge_id, data in edges.items() 
                             if data.get('from') == junction_id]
            
            print(f"Junction {junction_id} has {len(incoming_edges)} incoming and {len(outgoing_edges)} outgoing edges")
            
            # Create routes by connecting incoming to outgoing edges
            routes_created = 0
            for in_edge in incoming_edges:
                for out_edge in outgoing_edges:
                    # Verify if connection exists
                    if in_edge in connections and out_edge in connections.get(in_edge, []):
                        route = [in_edge, out_edge]
                        valid_routes.append({
                            'id': f'route_{len(valid_routes)}',
                            'edges': ' '.join(route)
                        })
                        routes_created += 1
                        if routes_created >= num_routes:
                            break
                if routes_created >= num_routes:
                    break
        
        # If we couldn't find direct routes, try to build simple routes
        if not valid_routes:
            print("No direct connections found, trying to find indirect routes...")
            # Use any connected sequences of edges that are at least 2 edges long
            for from_edge, to_edges in connections.items():
                for to_edge in to_edges:
                    if to_edge in connections:  # This edge connects to others
                        route = [from_edge, to_edge]
                        valid_routes.append({
                            'id': f'route_{len(valid_routes)}',
                            'edges': ' '.join(route)
                        })
                        if len(valid_routes) >= num_routes:
                            break
                if len(valid_routes) >= num_routes:
                    break
        
        if not valid_routes:
            # Last resort: just use single edges as routes
            print("No multi-edge routes found, using single edges...")
            for edge_id in edges:
                if not edge_id.startswith(':'):  # Skip internal edges
                    valid_routes.append({
                        'id': f'route_{len(valid_routes)}',
                        'edges': edge_id
                    })
                    if len(valid_routes) >= num_routes:
                        break
        
        print(f"Created {len(valid_routes)} valid routes")
        return valid_routes
        
    except Exception as e:
        print(f"Error analyzing network: {e}")
        return []

def create_route_file(route_file, routes, num_vehicles=50):
    """Create a route file with the given routes"""
    
    # Create XML
    root = ET.Element('routes')
    
    # Add vehicle type
    vtype = ET.SubElement(root, 'vType')
    vtype.set('id', 'car')
    vtype.set('accel', '2.6')
    vtype.set('decel', '4.5')
    vtype.set('sigma', '0.5')
    vtype.set('length', '5')
    vtype.set('maxSpeed', '15')
    
    # Add routes
    for route in routes:
        r = ET.SubElement(root, 'route')
        r.set('id', route['id'])
        r.set('edges', route['edges'])
    
    # Add vehicles
    for i in range(num_vehicles):
        route_idx = i % len(routes)
        v = ET.SubElement(root, 'vehicle')
        v.set('id', f'vehicle_{i}')
        v.set('type', 'car')
        v.set('route', routes[route_idx]['id'])
        v.set('depart', str(i * 10))  # Space vehicles 10 seconds apart
    
    # Write to file
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(route_file, 'w') as f:
        f.write(xml_str)
    
    print(f"Created route file with {len(routes)} routes and {num_vehicles} vehicles")

def main():
    if len(sys.argv) < 2:
        print("Usage: python improved_traffic.py <net_file> [<route_file>]")
        sys.exit(1)
    
    net_file = sys.argv[1]
    route_file = sys.argv[2] if len(sys.argv) > 2 else "simple_routes.xml"
    
    valid_routes = find_valid_routes(net_file)
    
    if not valid_routes:
        print("No valid routes found in the network!")
        sys.exit(1)
    
    create_route_file(route_file, valid_routes)
    
    print("\nNow run SUMO with this route file:")
    print(f"/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env/bin/sumo -n {net_file} -r {route_file} --stop-output stopinfo.xml --duration 1000")

if __name__ == "__main__":
    main()