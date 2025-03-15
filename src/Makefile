# 定义变量
PYTHON = python3
BSUB = bsub -q gpu -Is -gpu "num=1:mode=shared" 
CLIENTS_PER_GPU = 5 

SERVER_SCRIPT = create_server.py
CLIENT_SCRIPT = create_client.py
SRC_DIR = encrypted
NUM_CLIENTS = 50
MAX_CONCURRENT = 5
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
	# 按批次启动客户端
	for batch in $$(seq 0 $$(( ($(NUM_CLIENTS) + $(MAX_CONCURRENT) - 1) / $(MAX_CONCURRENT) - 1 )) ); do \
		echo "Starting batch $${batch}..."; \
		for i in $$(seq 0 $$(($(MAX_CONCURRENT) - 1))); do \
			client_id=$$((batch * $(MAX_CONCURRENT) + i)); \
			if [ $$client_id -lt $(NUM_CLIENTS) ]; then \
				echo "Launching client $${client_id}"; \
				if [ $$((client_id % $(CLIENTS_PER_GPU))) -eq 0 ]; then \
					$(BSUB) $(PYTHON) $(CLIENT_SCRIPT) --partition-id=$$client_id --client-number=$(NUM_CLIENTS) & \
				else \
					$(PYTHON) $(CLIENT_SCRIPT) --partition-id=$$client_id --client-number=$(NUM_CLIENTS) & \
				fi; \
			fi; \
		done; \
		echo "Waiting for batch $${batch} to initialize..."; \
		sleep 20; \
	done
	wait

# 清理 src 目录
clean:
	rm -rf $(SRC_DIR)/*
	rm -f server_address.txt
	rm -rf *.png
	rm -rf *.csv
	rm -rf *.txt

# 声明伪目标
.PHONY: server client clients clean plot