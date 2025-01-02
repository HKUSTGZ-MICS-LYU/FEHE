# 定义变量
PYTHON = python3
# BSUB = bsub -q gpu -Is -gpu "num=1"
BSUB = 
SERVER_SCRIPT = create_server.py
CLIENT_SCRIPT = create_client.py
SRC_DIR = encrypted
NUM_CLIENTS = 5  # 客户端数量

# 默认目标
all: 
	@echo "Please specify a target !!!"

# 运行 create_server.py
server:
	$(BSUB) $(PYTHON) $(SERVER_SCRIPT)

# 运行 create_client.py
client:
	$(BSUB) $(PYTHON) $(CLIENT_SCRIPT)

# 运行多个客户端
clients:
	for i in $$(seq 0 $(shell echo $$(($(NUM_CLIENTS) - 1)))); do \
		$(BSUB) $(PYTHON) $(CLIENT_SCRIPT) --partition-id=$$i --CLIENT_NUMER=$(NUM_CLIENTS) & \
	done
	wait  # 等待所有后台任务完成

# 清理 src 目录
clean:
	rm -rf $(SRC_DIR)/*

# 声明伪目标
.PHONY: server client clients clean