ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG C.UTF-8
ENV PASSWORD="123456" 
ENV XRDP_PORT=3389 SSH_PORT=22
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /root

# install python components
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ jupyter matplotlib numpy opencv-python pandas Pillow requests \
                scikit-learn jupyterlab  

# update jupyter config
RUN jupyter notebook --generate-config && mv /root/.jupyter/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py.bak
ADD ./jupyter_notebook_config.py /root/.jupyter/

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
ADD ./sources.list /etc/apt/sources.list

#install xrdp
RUN apt-get update --allow-unauthenticated && apt-get purge tightvncserver xrdp && \
    apt-get install -y xrdp xfce4 xfce4-goodies vnc4server tightvncserver vim curl net-tools iputils-ping firefox sudo supervisor wget python-pip python-dev gedit \
    software-properties-common aptitude locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy fcitx fcitx-googlepinyin openssh-server
RUN mkdir /var/run/sshd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

#config xrdp
RUN sed -i '/param7=/a\param8=-SecurityTypes' /etc/xrdp/sesman.ini
RUN sed -i '/param8=-SecurityTypes/a\param9=None' /etc/xrdp/sesman.ini

# install ssh
RUN mkdir /root/.ssh
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "xfce4-session" > ~/.xsession

#set env config(GPU)
RUN echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/sbin:$PATH' >> /root/.bashrc

WORKDIR /root/.init
RUN mkdir jupyterlabspace
WORKDIR /root/.init/jupyterlabspace
EXPOSE $XRDP_PORT $SSH_PORT
RUN find / -name "*.pyc" | xargs rm -rf

RUN apt-get update --allow-unauthenticated && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  && \ 
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \ 
 conda config --set show_channel_urls yes

# Install gdal
RUN conda install -y gdal=2.2.2

# Install MMCV
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# Install MMDetection
RUN conda clean --all
WORKDIR /root
RUN git clone -b building https://github.com/abcxs/building_extraction.git mmdetection
WORKDIR /root/mmdetection
RUN git checkout building && git pull
ENV FORCE_CUDA="1"
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements/build.txt && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --no-cache-dir -e .
WORKDIR /root/mmdetection/mmdet/rotation_libs
RUN python setup.py build_ext --inplace

WORKDIR /root
# install startup.sh
RUN mkdir -p /root/.init
ADD ./startup.sh /root/.init
ADD ./config.sh /root/.init
ADD ./tif_process_aios.sh /root/tif_process.sh
RUN chmod a+x /root/.init/startup.sh && chmod a+x /root/.init/config.sh && chmod a+x /root/tif_process.sh
ENTRYPOINT ["/root/.init/startup.sh"]
