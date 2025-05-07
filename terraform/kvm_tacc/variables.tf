variable "suffix" {
  description = "Suffix for resource names (use net ID)"
  type        = string
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
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
    "node3" = "192.168.1.13"
  }
}

variable "OS_APP_CRED_ID" {
  description = "The OpenStack application credential ID"
  type        = string
}

variable "OS_APP_CRED_SECRET" {
  description = "The OpenStack application credential secret"
  type        = string
}
