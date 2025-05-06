data "openstack_networking_network_v2" "sharednet1" {
  name = "sharednet1"
}

data "openstack_networking_subnet_v2" "sharednet1_subnet" {
  name = "sharednet1-subnet"
}

data "openstack_networking_secgroup_v2" "allow_ssh" {
  name = "allow-ssh"
}

data "openstack_networking_secgroup_v2" "allow_30000_32767" {
  name = "nodeport-30000-32767"
}

#data "openstack_networking_secgroup_v2" "allow_8000" {
#  name = "allow-8000"
#}
#
#data "openstack_networking_secgroup_v2" "allow_8080" {
#  name = "allow-8080"
#}
#
#data "openstack_networking_secgroup_v2" "allow_8081" {
#  name = "allow-8081"
#}
#
#data "openstack_networking_secgroup_v2" "allow_http_80" {
#  name = "allow-http-80"
#}
#
#data "openstack_networking_secgroup_v2" "allow_9090" {
#  name = "allow-9090"
#}

