source mypython/bin/activate
pip install opencv-contrib-python
python -m pip install tensorflow==1.14
pip install cython; pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
cd detectron2
sudo MACOSX_DEPLOYMENT_TARGET=10.15 CC=clang CXX=clang++ pip3 install -e .
cd ../
