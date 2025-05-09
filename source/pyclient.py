import sys
import driver
import socket
import argparse

if __name__ == '__main__':
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR',
                        help='Bot ID (default: SCR)')

    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                        help='Maximum number of learning episodes (default: 1)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                        help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default=None,
                        help='Name of the track')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
    parser.add_argument('--car', action='store', dest='car', default=None,
                        help='Stage (0 - Toyota, 1 - Peugeot, 2 - Mitsubishi)')

    arguments = parser.parse_args()

    # Print summary
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
    except socket.error as msg:
        print('Could not make a socket. Exiting...')
        sys.exit(-1)

    # Increase timeout to prevent frequent disconnections
    sock.settimeout(5.0)

    shutdownClient = False
    curEpisode = 0

    verbose = False

    d = driver.Driver(arguments.stage)

    while not shutdownClient:
        while True:
            print('Sending id to server: ', arguments.id)
            buf = arguments.id + d.init()
            print('Sending init string to server:', buf)

            try:
                sock.sendto(buf.encode('utf-8'), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)

            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')  # Explicitly decode with utf-8
            except socket.timeout:
                print("Didn't get response from server... Retrying")
                continue
            except socket.error as msg:
                print("Socket error:", msg)
                continue

            if '***identified***' in buf:
                print('Received:', buf)
                break

        currentStep = 0

        while True:
            try:
                sock.settimeout(10)  # Allow more time for response
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')  # Explicitly decode with utf-8
            except socket.timeout:
                print("Didn't get response from server... Retrying")
                buf = None
            except socket.error as msg:
                print("Socket error:", msg)
                buf = None

            if verbose and buf:
                print('Received:', buf)

            if buf and '***shutdown***' in buf:
                d.onShutDown()
                shutdownClient = True
                print('Client Shutdown')
                break

            if buf and '***restart***' in buf:
                d.onRestart()
                print('Client Restart')
                break

            currentStep += 1
            if currentStep != arguments.max_steps:
                if buf:
                    buf = d.drive(buf, arguments.track, arguments.car) 
                else:
                    buf = "(meta 1)"  # Prevent timeout when no input

            if verbose:
                print('Sending:', buf)

            if buf:
                try:
                    sock.sendto(buf.encode('utf-8'), (arguments.host_ip, arguments.host_port))
                except socket.error as msg:
                    print("Failed to send data...Exiting...")
                    sys.exit(-1)

        curEpisode += 1
        if curEpisode == arguments.max_episodes:
            shutdownClient = True

    sock.close()