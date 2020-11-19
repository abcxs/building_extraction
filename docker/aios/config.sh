#!/bin/bash
set -e 
im-config -n fcitx
cat>>~/.vimrc<<EOF
set fenc=utf-8
set fencs=utf-8,usc-bom,euc-jp,gb18030,gbk,gb2312,cp936,bigï¼?
set enc=utf-8
let &termencoding=&encoding
EOF
cat > ~/.xprofile << EOF
export LANG=zh_CN.UTF-8
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS="@im=fcitx"
fcitx -d 
EOF
cat > /etc/profile << EOF
export  LANG=C.UTF-8
EOF
source /etc/profile