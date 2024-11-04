FROM mlnotebook

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

RUN apt update && \
    apt-get install -y  --no-install-recommends \
        tmux less sudo nano xterm git \
        swig sudo mesa-utils libgl1-mesa-glx

# User: robot (password: robot) with sudo power

RUN useradd -ms /bin/bash robot && echo "robot:robot" | chpasswd
RUN usermod -u $UID robot && groupmod -g $GID robot

RUN adduser robot sudo  && \
    adduser robot audio  && \
    adduser robot video  && \
    adduser robot dialout

RUN chown -R robot.robot /opt

USER robot


ADD requirements.txt /opt/requirements.txt

ENV PATH=$PATH:/home/robot/.local/bin

RUN pip install -r  /opt/requirements.txt

RUN echo "set -g mouse on" > /$HOME/.tmux.conf 

WORKDIR /opt/gym

CMD /usr/bin/tmux



