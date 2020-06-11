
install:
	cd ~ && git clone https://github.com/mkowiel/riconv.git
	mkdir -p ~/data
	sudo nvidia-smi -ac 5001,1590
	ln -s /home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.so.1 /home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/tensorflow_core/libtensorflow_framework.so
	conda config --append channels conda-forge
	conda install -n tensorflow_p36 transforms3d
	. activate tensorflow_p36 && cd ./tf_ops/3d_interpolation && bash tf_interpolate_compile.sh
	. activate tensorflow_p36 && cd ./tf_ops/grouping && bash tf_grouping_compile.sh
	. activate tensorflow_p36 && cd ./tf_ops/sampling && bash tf_sampling_compile.sh

train:
	. activate tensorflow_p36 && python3 train_val_cls.py
