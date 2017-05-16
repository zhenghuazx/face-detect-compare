# deploy server
install anaconda from https://www.continuum.io/downloads
install django pip install Django==1.8
conda install -c menpo dlib=19.4
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
pip install keras
pip install keras_vggface
# install git
sudo yum -y install git-all
# Install gcc-4.8/make and other development tools on Amazon Linux
# Reference: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compile-software.html
# Install Python, Numpy, Scipy and set up tools.
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python27 python27-setuptools python27-tools python-pip graphviz

conda install -c conda-forge pylibmc=1.5.2
pip install 'git+git://github.com/dlrust/python-memcached-stats.git'

pip install psycopg2
#install memecached on EC2 Amazon Linux; try following either one
sudo yum install libevent-devel
sudo yum install memcached
# start memcached
sudo service memcached start
# stop memcached
sudo service memcached stop