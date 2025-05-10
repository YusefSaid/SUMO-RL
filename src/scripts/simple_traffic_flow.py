#!/usr/bin/env python3
"""
Generate a simple flow of vehicles that will encounter traffic lights
in the existing network.
"""
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def find_traffic_light_edges(net_file):
    """Find edges connected to traffic light controlled intersections"""
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    # Find traffic light junctions
    tl_junctions = set()
    for junction in root.findall('junction'):
        if junction.get('type') == 'traffic_light':
            tl_junctions.add(junction.get('id'))
    
    print(f"Found {len(tl_junctions)} traffic light junctions: {', '.join(tl_junctions)}")
    
    # Find incoming edges to these junctions
    incoming_edges = []
    outgoing_edges = []
    
    for connection in root.findall('connection'):
        to_junction = None
        from_junction = None
        
        # Find the junction this connection leads to
        for edge in root.findall('edge'):
            if edge.get('id') == connection.get('to'):
                to_junction = edge.get('to')
            if edge.get('id') == connection.get('from'):
                from_junction = edge.get('from')
        
        if to_junction in tl_junctions and connection.get('from') not in incoming_edges:
            incoming_edges.append(connection.get('from'))
        
        if from_junction in tl_junctions and connection.get('to') not in outgoing_edges:
            outgoing_edges.append(connection.get('to'))
    
    print(f"Found {len(incoming_edges)} edges leading to traffic lights")
    print(f"Found {len(outgoing_edges)} edges leading from traffic lights")
    
    return incoming_edges, outgoing_edges

def create_route_file(route_file, incoming_edges, outgoing_edges, num_vehicles=100):
    """Create a route file with routes that pass through traffic lights"""
    
    # Create viable routes
    routes = []
    for i, in_edge in enumerate(incoming_edges[:5]):  # Limit to first 5 incoming edges
        for j, out_edge in enumerate(outgoing_edges[:5]):  # Limit to first 5 outgoing edges
            routes.append({
                'id': f'route_{i}_{j}',
                'edges': f'{in_edge} {out_edge}'
            })
    
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
        v.set('depart', str(i * 5))  # Space vehicles 5 seconds apart
    
    # Write to file
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(route_file, 'w') as f:
        f.write(xml_str)
    
    print(f"Created route file with {len(routes)} routes and {num_vehicles} vehicles")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_traffic.py <net_file> [<route_file>]")
        sys.exit(1)
    
    net_file = sys.argv[1]
    route_file = sys.argv[2] if len(sys.argv) > 2 else "auto_routes.xml"
    
    incoming_edges, outgoing_edges = find_traffic_light_edges(net_file)
    
    if not incoming_edges or not outgoing_edges:
        print("No traffic light edges found in the network!")
        sys.exit(1)
    
    create_route_file(route_file, incoming_edges, outgoing_edges)
    
    print("\nNow run SUMO with this route file:")
    print(f"sumo -n {net_file} -r {route_file} --stop-output stopinfo.xml --duration 1000")

if __name__ == "__main__":
    main()