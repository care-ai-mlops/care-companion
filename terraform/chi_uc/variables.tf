variable "suffix" {
  description = "Suffix for resource names (project ID)"
  type        = string
  nullable = false
  default = "project51"
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
  }
}

variable "reservation_id" {
    description = "Reservation ID"
    type = string
    nullable = false
}
