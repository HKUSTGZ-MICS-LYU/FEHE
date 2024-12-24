# 定义变量
PYTHON = python3
BSUB = bsub -q gpu -Is -gpu "num=1"
SERVER_SCRIPT = create_server.py
CLIENT_SCRIPT = create_client.py
SRC_DIR = encrypted

# 默认目标
all: 
	@echo "Please specify a target !!!"

# 运行 create_server.py
server:
	$(BSUB) $(PYTHON) $(SERVER_SCRIPT)

# 运行 create_client.py
client:
	$(BSUB) $(PYTHON) $(CLIENT_SCRIPT)

# 清理 src 目录
clean:
	rm -rf $(SRC_DIR)/*

# 声明伪目标
.PHONY: server client clean