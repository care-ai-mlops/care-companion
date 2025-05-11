# First set appropriate IP addresses in the bashrc file
# echo 'export CHI_FLOATING_IP=IP_ADDRESS' >> ~/.bashrc && echo 'export KVM_FLOATING_IP=IP_ADDRESS' >> ~/.bashrc
envsubst < care-companion/Docker/prometheus.template.yml > care-companion/Docker/prometheus.yml