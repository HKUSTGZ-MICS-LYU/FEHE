from socket import *
import json
import random
import struct
import time

# Constants
POLY_DEGREE = 4096
PLAINTEXT_MODULUS = 12289  # Should match FPGA's modulus
BUFFER_SIZE = 4096 * 4

def send_numbers(sock, numbers):
    """Send numbers list with length header"""
    data = json.dumps(numbers).encode('utf-8')
    sock.send(struct.pack('!I', len(data)))  # Send data length
    sock.send(data)                          # Send JSON data

def receive_numbers(sock):
    """Receive numbers list with length header"""
    length_data = sock.recv(4)
    if not length_data:
        return None
    data_length = struct.unpack('!I', length_data)[0]
    
    received = bytearray()
    while len(received) < data_length:
        remaining = data_length - len(received)
        received += sock.recv(min(remaining, BUFFER_SIZE))
    
    return json.loads(received.decode('utf-8'))

def main():
    seed = time.time()
    server_address = ('192.168.2.99', 123)
    
    with socket(AF_INET, SOCK_STREAM) as sock:
        sock.connect(server_address)
        
        try:
            while True:
                # Generate random numbers for encryption
                numbers = [random.randint(0, PLAINTEXT_MODULUS-1) 
                          for _ in range(POLY_DEGREE)]
                
                # Send numbers to FPGA
                send_numbers(sock, numbers)
                print(f"Sent {len(numbers)} numbers for encryption")
                
                # Receive encrypted result
                encrypted_result = receive_numbers(sock)
                if encrypted_result:
                    print(f"Received encrypted result (first 10): {encrypted_result[:10]}")
                else:
                    print("Connection closed")
                    break
                    
        except KeyboardInterrupt:
            print("\nClient terminated")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()