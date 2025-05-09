import socket

port = 3001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind(("localhost", port))
    print(f"Port {port} is free and can be used!")
    s.close()
except OSError as e:
    print(f"Port {port} is in use: {e}")
