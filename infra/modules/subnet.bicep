// Adds a dedicated subnet for the Container Apps Environment into an
// existing virtual network (typically the AKS-managed vnet).
// Required delegation: Microsoft.App/environments.
//
// Deploy this module scoped to the resource group that owns the vnet.
param vnetName string
param subnetName string
param addressPrefix string

resource vnet 'Microsoft.Network/virtualNetworks@2024-01-01' existing = {
  name: vnetName
}

resource subnet 'Microsoft.Network/virtualNetworks/subnets@2024-01-01' = {
  parent: vnet
  name: subnetName
  properties: {
    addressPrefix: addressPrefix
    delegations: [
      {
        name: 'aca-delegation'
        properties: {
          serviceName: 'Microsoft.App/environments'
        }
      }
    ]
    privateEndpointNetworkPolicies: 'Disabled'
    privateLinkServiceNetworkPolicies: 'Enabled'
  }
}

output id string = subnet.id
output name string = subnet.name
