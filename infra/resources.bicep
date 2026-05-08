// Resource-group-scoped orchestrator. Creates only the Container Apps
// Environment (vnet-attached), the user-assigned MI, the Container App
// itself, and a dedicated subnet in the existing AKS vnet.
//
// ACR (jianshn) and storage account (benchmarkdatangjason) are
// referenced as existing resources (managed outside this template);
// only RBAC role assignments for the new MI are created against them.
targetScope = 'resourceGroup'

param location string
param tags object

param containerAppName string
param containerImage string

@description('Name of the existing ACR used for image pulls.')
param acrName string

@description('Resource group of the existing ACR (may be in a different RG / same subscription).')
param acrResourceGroup string

@description('Name of the existing storage account used by benchmark_storage.py.')
param storageAccountName string

@description('Resource group of the existing storage account.')
param storageAccountResourceGroup string

@description('Blob container name used by benchmark_storage.py.')
param storageContainerName string

param entraClientId string
param entraTenantId string

@description('Base64-encoded kubeconfig (kubelogin MSI mode).')
@secure()
param kubeconfigB64 string

@description('Hugging Face access token.')
@secure()
param hfToken string

@description('Existing Azure AI / OpenAI account name to grant the MI access to.')
param openAiAccountName string

@description('Resource group of the existing Azure AI / OpenAI account.')
param openAiResourceGroup string

@description('Name of the existing virtual network to attach the Container Apps Environment to.')
param vnetName string

@description('Resource group containing the existing virtual network.')
param vnetResourceGroup string

@description('Name of the new subnet to create for the Container Apps Environment.')
param containerAppSubnetName string

@description('Address prefix for the new Container Apps subnet. Must lie inside the vnet, not overlap existing subnets, and be at least /27.')
param containerAppSubnetPrefix string

@description('If true, the Container Apps Environment ingress is internal-only.')
param containerAppEnvInternal bool

// ---------- Existing resources (lookups for derived values) ----------
resource existingAcr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' existing = {
  name: acrName
  scope: resourceGroup(acrResourceGroup)
}

// ---------- Subnet in existing vnet (cross-RG) ----------
module subnet 'modules/subnet.bicep' = {
  name: 'aca-subnet'
  scope: resourceGroup(vnetResourceGroup)
  params: {
    vnetName: vnetName
    subnetName: containerAppSubnetName
    addressPrefix: containerAppSubnetPrefix
  }
}

// ---------- Container Apps Environment ----------
module env 'modules/containerAppEnv.bicep' = {
  name: 'cae'
  params: {
    name: '${containerAppName}-env'
    location: location
    tags: tags
    infrastructureSubnetId: subnet.outputs.id
    internal: containerAppEnvInternal
  }
}

// ---------- User-assigned managed identity for the Container App ----------
resource appIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${containerAppName}-mi'
  location: location
  tags: tags
}

// ---------- Role assignments on existing ACR + storage (cross-RG) ----------
module acrRole 'modules/acrPullRole.bicep' = {
  name: 'role-acrpull'
  scope: resourceGroup(acrResourceGroup)
  params: {
    acrName: acrName
    principalId: appIdentity.properties.principalId
  }
}

module storageRole 'modules/storageBlobRole.bicep' = {
  name: 'role-storageblob'
  scope: resourceGroup(storageAccountResourceGroup)
  params: {
    storageAccountName: storageAccountName
    principalId: appIdentity.properties.principalId
  }
}

module openAiRole 'modules/openAiRole.bicep' = {
  name: 'role-openai'
  scope: resourceGroup(openAiResourceGroup)
  params: {
    accountName: openAiAccountName
    principalId: appIdentity.properties.principalId
  }
}

// ---------- Container App ----------
module containerApp 'modules/containerApp.bicep' = {
  name: 'containerApp'
  params: {
    name: containerAppName
    location: location
    tags: tags
    environmentId: env.outputs.id
    userAssignedIdentityId: appIdentity.id
    userAssignedIdentityClientId: appIdentity.properties.clientId
    containerImage: containerImage
    acrLoginServer: existingAcr.properties.loginServer
    storageAccountName: storageAccountName
    storageContainerName: storageContainerName
    entraClientId: entraClientId
    entraTenantId: entraTenantId
    kubeconfigB64: kubeconfigB64
    hfToken: hfToken
  }
  dependsOn: [
    acrRole
    storageRole
    openAiRole
  ]
}

output containerAppFqdn string = containerApp.outputs.fqdn
output containerAppPrincipalId string = appIdentity.properties.principalId
output acrLoginServer string = existingAcr.properties.loginServer
