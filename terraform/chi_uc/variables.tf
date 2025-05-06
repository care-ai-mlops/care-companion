variable "suffix" {
  description = "Suffix for resource names (project ID)"
  type        = string
  nullable = false
  default = "project51"
}

variable "reservation_token" {
    description = "Reservation token"
    type = string
    nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "key1"
}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.10"
#    "node2" = "192.168.1.12"
#    "node3" = "192.168.1.13"
  }
}
