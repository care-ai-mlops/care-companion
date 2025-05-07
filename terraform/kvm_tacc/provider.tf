provider "openstack" {
  auth_url                    = "https://kvm.tacc.chameleoncloud.org:5000"
  region                      = "KVM@TACC"
  interface                   = "public"
  identity_api_version        = 3
  auth_type                   = "v3applicationcredential"

  application_credential_id     = var.OS_APP_CRED_ID
  application_credential_secret = var.OS_APP_CRED_SECRET
}
