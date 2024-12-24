import base64

def write_data(file_name: str, data):
    # Input List of data, Output file with data list
    for i in range(len(data)):
        data[i] = base64.b64encode(data[i])
    with open(file_name, 'wb') as f:
        for d in data:
            f.write(d)
    f.close()
    
        
def read_data(file_name: str) -> bytes:
    data_list = []
    with open(file_name, 'rb') as f:
        for line in f:
            data = base64.b64decode(line)
            data_list.append(data)
    f.close()
    #base64 to bytes
    return data_list