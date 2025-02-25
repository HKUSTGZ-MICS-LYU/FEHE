# 定义变量
PYTHON = python3
BSUB = bsub -q gpu -Is -gpu "num=1" 



SERVER_SCRIPT = create_server.py
CLIENT_SCRIPT = create_client.py
SRC_DIR = encrypted
NUM_CLIENTS = 50
# 默认目标
all: 
	@echo "Please specify a target !!!"

plot: 
	$(BSUB) $(PYTHON) utils/plot_accuracy.py
	$(BSUB) $(PYTHON) utils/plot_time_stat.py
	
# 运行 create_server.py
server:
	rm -f server_address.txt  encrypted/*
	$(BSUB) $(PYTHON) $(SERVER_SCRIPT)

# 运行 create_client.py
client:
	while [ ! -f server_address.txt ]; do sleep 1; done
	$(BSUB) $(PYTHON) $(CLIENT_SCRIPT)

# 运行多个客户端
clients:
	while [ ! -f server_address.txt ]; do sleep 1; done
	for i in $$(seq 0 $(shell echo $$(($(NUM_CLIENTS) - 1)))); do \
		$(BSUB) $(PYTHON) $(CLIENT_SCRIPT) --partition-id=$$i --client-number=$(NUM_CLIENTS) & \
	done
	wait  # 等待所有后台任务完成

# 清理 src 目录
clean:
	rm -rf $(SRC_DIR)/*
	rm -f server_address.txt
	rm -rf *.png
	rm -rf *.csv
	rm -rf *.txt

# 声明伪目标
.PHONY: server client clients clean plot