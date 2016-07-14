FROM dmlc/mxnet:cuda

RUN wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip
RUN unzip -u cifar10.zip
RUN rm cifar10.zip

RUN git clone https://github.com/shenfei/gpu_test.git
