
#docker run --runtime=nvidia --rm -ti -v "${PWD}:/app" tensorflow/tensorflow:latest-gpu-jupyter
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
#docker image prune -a
docker image build -t tf_gpu_custom .
#docker run --runtime=nvidia --detach --name lsh_gpu_environment tf_gpu_custom
docker run --runtime=nvidia --rm -ti --name lsh_gpu_environment tf_gpu_custom