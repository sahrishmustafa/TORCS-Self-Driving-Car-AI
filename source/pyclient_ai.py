import sys
import socket
import argparse

from msgParser import MsgParser
from aidriver import AIDriver

KEY_MAP = {
    'angle': 'angle',
    'curLapTime ': 'curlaptime',
    'damage': 'damage',
    'distFromStart ': 'distfromstart',
    'distRaced ': 'distraced',
    'fuel': 'fuel',
    'gear': 'gear',
    'lastLapTime ': 'lastlaptime',
    'racePos ': 'racepos',
    'rpm': 'rpm',
    'speedX': 'speedx',
    'speedY': 'speedy',
    'speedZ': 'speedz',
    'track': 'track',
    'trackPos': 'trackpos',
    'wheelSpinVel ': 'wheelspinvel',
    'z': 'z',
    'focus': 'focus'
}

def parse_sensor_data(data_str):
    """Convert TORCS protocol string to properly formatted sensor data"""
    import re
    from collections import defaultdict

    # Use lowercase keys from the start
    temp_data = defaultdict(float, {
        'speedx': 0.0,
        'speedy': 0.0,
        'speedz': 0.0,
        'rpm': 0.0,
        'angle': 0.0,
        'trackpos': 0.0,
        'fuel': 0.0,
        'curlaptime': 0.0,
        'lastlaptime': 0.0,
        'distfromstart': 0.0,
        'distraced': 0.0,
        'damage': 0.0,
        'racepos': 1.0,
        'z': 0.0,
        'focus': [0.0] * 5,
        'wheelspinvel': [0.0] * 4,
        'track': [0.0] * 19
    })

    pattern = re.findall(r'\((\w+)\s+([^)]+)\)', data_str)
    for key, value in pattern:
        key = key.strip().lower()  # Ensure clean, lowercase keys
        mapped_key = KEY_MAP.get(key, key)

        if mapped_key in ['focus', 'wheelspinvel', 'track']:
            values = [float(v) for v in value.strip().split()]
            max_len = {
                'focus': 5,
                'wheelspinvel': 4,
                'track': 19
            }[mapped_key]
            temp_data[mapped_key] = values[:max_len] + [0.0] * (max_len - len(values))
        else:
            try:
                temp_data[mapped_key] = float(value)
            except ValueError:
                continue

    # Return fully lowercased keys for downstream model
    sensor_data = {}
    for k, v in temp_data.items():
        sensor_data[k] = v if isinstance(v, list) else float(v)

    return sensor_data

def send_initial_message(sock, host_ip, host_port, bot_id):
    """Send initial bot ID to server and receive confirmation"""
    print('Sending id to server:', bot_id)
    buf = bot_id
    print('Sending init string to server:', buf)

    try:
        sock.sendto(buf.encode('utf-8'), (host_ip, host_port))
    except socket.error as msg:
        print("Failed to send data...Exiting...")
        sys.exit(-1)

    try:
        buf, addr = sock.recvfrom(1000)
        buf = buf.decode('utf-8')  # Explicitly decode with utf-8
    except socket.timeout:
        print("Didn't get response from server... Retrying")
        return None
    except socket.error as msg:
        print("Socket error:", msg)
        return None

    if '***identified***' in buf:
        print('Received:', buf)
        return True
    return False

def receive_data_from_server(sock):
    """Receive data with aggressive retry logic"""
    while True:  # Loop indefinitely until data is received
        try:
            sock.settimeout(0.5)  # Very short timeout to avoid blocking
            buf, addr = sock.recvfrom(1000)
            return buf.decode('utf-8')
        except socket.timeout:
            continue  # Silently retry
        except socket.error as msg:
            print(f"Fatal socket error: {msg}")
            sys.exit(-1)

def drive(sensor_data, ai_driver):
    """Compute the controls based on sensor data using AI driver"""
    controls = ai_driver.get_control(sensor_data)

    return {
        'accel': controls['accel'],   # 0-1
        'brake': controls['brake'],   # 0-1
        'steer': controls['steer'],   # -1 to 1
        'gear': controls['gear'],     # Integer 1-6
        'focus': [0, 0, 0, 0, 0],      # Not used but required
        'meta': 0
    }

def run_episode(sock, ai_driver, arguments):
    """Run a single episode of the simulation"""
    currentStep = 0
    parser = MsgParser()
    
    while True:
        # 1. Receive sensor data from server
        buf = None
        while not buf:  # Keep trying until data arrives
            buf = receive_data_from_server(sock)
        
        # 2. Parse data and compute controls
        sensor_dict = parse_sensor_data(buf)
        control_dict = drive(sensor_dict, ai_driver)
        control_msg = parser.dict_to_msg(control_dict)
        
        # 3. Send control message back
        try:
            sock.sendto(control_msg.encode('utf-8'), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print(f"Critical send error: {msg}")
            sys.exit(-1)
        
        currentStep += 1
        if arguments.max_steps > 0 and currentStep >= arguments.max_steps:
            break

def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
    parser.add_argument('--host', action='store', dest='host_ip', default='localhost', help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001, help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR', help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1, help='Maximum number of learning episodes (default: 1)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0, help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default=None, help='Name of the track')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3, help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
    parser.add_argument('--car', action='store', dest='car', default=None, help='Stage (0 - Toyota, 1 - Peugeot, 2 - Mitsubishi)')
    arguments = parser.parse_args()

    print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
    print('Bot ID:', arguments.id)
    print('Maximum episodes:', arguments.max_episodes)
    print('Maximum steps:', arguments.max_steps)
    print('Track:', arguments.track)
    print('Stage:', arguments.stage)
    print('Car:', arguments.car)
    print('*********************************************')

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
    except socket.error as msg:
        print('Could not make a socket. Exiting...')
        sys.exit(-1)

    # sock.settimeout(5.0)
    shutdownClient = False
    curEpisode = 0
    ai_driver = AIDriver(arguments.track, arguments.car)

    # Send bot ID to server
    while not shutdownClient:
        if send_initial_message(sock, arguments.host_ip, arguments.host_port, arguments.id):
            break

    # Run simulation episodes
    while not shutdownClient and curEpisode < arguments.max_episodes:
        # run_episode(sock, ai_driver, arguments)
        run_episode(sock, ai_driver, arguments)
        curEpisode += 1

    # shutdown_and_cleanup(sock)

if __name__ == '__main__':
    main()