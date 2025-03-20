from socket import *
import json
import random
import struct
import time

# Constants
POLY_DEGREE = 4096
PLAINTEXT_MODULUS = 12289  # Should match FPGA's modulus
BUFFER_SIZE = 4096 * 3
MAX_REQUESTS = 100  # 设置最大请求次数
RETRY_DELAY = 2     # 重试间隔(秒)
MAX_RETRIES = 3     # 最大重试次数

def send_numbers(sock, numbers):
    """Send numbers list with length header"""
    data = json.dumps(numbers).encode('utf-8')
    sock.send(struct.pack('!I', len(data)))  # Send data length
    sock.send(data)                          # Send JSON data

def receive_numbers(sock):
    """Receive numbers list with length header"""
    try:
        length_data = sock.recv(4)
        if not length_data:
            return None
        data_length = struct.unpack('!I', length_data)[0]
        
        received = bytearray()
        while len(received) < data_length:
            remaining = data_length - len(received)
            chunk = sock.recv(min(remaining, BUFFER_SIZE))
            if not chunk:  # 连接关闭
                return None
            received += chunk
        
        return json.loads(received.decode('utf-8'))
    except Exception as e:
        print(f"接收数据错误: {str(e)}")
        return None

def process_single_request(server_address):
    """处理单个请求，返回(成功标志, 耗时)"""
    # 创建新连接
    with socket(AF_INET, SOCK_STREAM) as sock:
        try:
            # 设置超时
            sock.settimeout(10)
            sock.connect(server_address)
            
            start_time = time.time()
            
            # 生成随机数
            numbers = [random.randint(0, PLAINTEXT_MODULUS-1) 
                      for _ in range(POLY_DEGREE)]
            print(f"Generated {len(numbers)} numbers for encryption")
            print(f"First 10 numbers: {numbers[:10]}")
            
            # 发送数据
            send_numbers(sock, numbers)
            print(f"Sent {len(numbers)} numbers for encryption")
            
            # 接收加密结果
            encrypted_result = receive_numbers(sock)
            elapsed = time.time() - start_time
            
            if encrypted_result:
                print(f"Received encrypted result (first 10): {encrypted_result[:10]}")
                print(f"Request completed in {elapsed:.4f} seconds")
                return True, elapsed
            else:
                print(f"No response or invalid response. Time elapsed: {elapsed:.4f} seconds")
                return False, elapsed
                
        except ConnectionRefusedError:
            print("Server connection refused")
            return False, 0
        except Exception as e:
            print(f"Request error: {str(e)}")
            return False, 0

def main():
    random.seed(time.time())
    server_address = ('192.168.2.99', 123)
    
    # 统计变量
    request_count = 0
    success_count = 0
    total_time = 0
    failed_requests = 0
    start_total = time.time()
    
    try:
        while request_count < MAX_REQUESTS:
            request_count += 1
            print(f"\n--- Request {request_count}/{MAX_REQUESTS} ---")
            
            # 执行请求，有重试机制
            retry_count = 0
            success = False
            
            while not success and retry_count <= MAX_RETRIES:
                if retry_count > 0:
                    print(f"重试 #{retry_count}...")
                    time.sleep(RETRY_DELAY)
                
                success, elapsed = process_single_request(server_address)
                
                if success:
                    success_count += 1
                    total_time += elapsed
                    break
                
                retry_count += 1
            
            if not success:
                failed_requests += 1
                print(f"请求 #{request_count} 在 {MAX_RETRIES} 次重试后失败")
                
            # 在请求之间稍微延迟，避免服务器过载
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n客户端被用户终止")
    finally:
        # 输出统计信息
        total_elapsed = time.time() - start_total
        print("\n===== 性能统计 =====")
        print(f"总请求数: {request_count}")
        print(f"成功请求数: {success_count}")
        print(f"失败请求数: {failed_requests}")
        print(f"总耗时: {total_elapsed:.4f} 秒")
        if success_count > 0:
            print(f"平均每次请求时间: {total_time/success_count:.4f} 秒")
            print(f"每秒请求数: {success_count/total_elapsed:.2f}")
        print("=====================")

if __name__ == "__main__":
    main()