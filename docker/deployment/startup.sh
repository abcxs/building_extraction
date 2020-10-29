#!/bin/bash
echo "root:$PASSWORD" | chpasswd
#jupyter notebook --allow-root &
#start xrdp
rm -f /run/xrdp/xrdp*
/root/.init/config.sh &

#config nvidia
source /root/.bashrc
ldconfig

#/etc/init.d/xrdp restart
/usr/sbin/xrdp
/usr/sbin/xrdp-sesman


#start sshd
/usr/sbin/sshd -D &
jupyter lab --allow-root --ip 0.0.0.0
