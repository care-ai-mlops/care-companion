

resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each   = var.nodes
    name       = "sharednet1-${each.key}-mlops-${var.suffix}"
    network_id = data.openstack_networking_network_v2.sharednet1.id
    security_group_ids = [
      data.openstack_networking_secgroup_v2.allow_ssh.id,
    ]
}

# Create the instance, referencing the baremetal flavor, and scheduler hint
resource "openstack_compute_instance_v2" "node1" {

  for_each   = var.nodes

  name = "${each.key}-mlops-${var.suffix}"
  image_name = "CC-Ubuntu24.04-CUDA"
  flavor_name = "baremetal"
  key_pair = "key1"

  network {
    port = openstack_networking_port_v2.sharednet1_ports[each.key].id
  }

  scheduler_hints {
    additional_properties = {
        "reservation" = "${var.reservation_id}"
    }
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
  
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet1_ports["node1"].id
}
