// Subscription-scoped entry point.
// Creates (or reuses) the `gpu-benchmark` resource group in centralus
// and deploys all child resources via the resources module.
targetScope = 'subscription'

@description('Resource group name for all resources.')
param resourceGroupName string = 'gpu-benchmark'

@description('Azure region for all resources.')
param location string = 'centralus'

@description('Location of the resource group itself (metadata only). Must match the existing RG location if it already exists. Defaults to `location`.')
param resourceGroupLocation string = location

@description('Container App name.')
param containerAppName string = 'automated-gpu-benchmark'

@description('Storage account name (existing) used by benchmark_storage.py.')
param storageAccountName string = 'benchmarkdatangjason'

@description('Resource group of the existing storage account.')
param storageAccountResourceGroup string = 'gpu-benchmark'

@description('Blob container name used by benchmark_storage.py.')
param storageContainerName string = 'runs'

@description('Existing Azure Container Registry name used for image pulls.')
param acrName string = 'jianshn'

@description('Resource group of the existing ACR.')
param acrResourceGroup string = 'container-rg'

@description('Container image reference (must live in the ACR above).')
param containerImage string = 'jianshn.azurecr.io/llm-sizing:latest'

@description('Application (client) ID of the existing Entra ID enterprise app llm-vram-calculator-auth. Auth uses a federated credential on this app, no client secret is required.')
param entraClientId string

@description('Entra tenant ID. Defaults to current subscription tenant.')
param entraTenantId string = subscription().tenantId

@description('Base64-encoded kubeconfig (kubelogin MSI mode). Exposed inside the container as env var KUBECONFIG_B64.')
@secure()
param kubeconfigB64 string

@description('Hugging Face access token. Exposed inside the container as env var HF_TOKEN.')
@secure()
param hfToken string

@description('Existing Azure AI / OpenAI account name to grant the MI access to.')
param openAiAccountName string = 'jianshn-eastus2-foundry'

@description('Resource group of the existing Azure AI / OpenAI account.')
param openAiResourceGroup string = 'jason-open-ai'

@description('Existing AKS cluster the container app will run benchmark workloads on.')
param aksClusterName string = 'multi-gpu-benchmark'

@description('Resource group of the existing AKS cluster.')
param aksResourceGroup string = 'gpu-benchmark'

@description('Existing virtual network the Container Apps Environment will attach to.')
param vnetName string = 'aks-vnet-21444232'

@description('Resource group of the existing virtual network (the AKS node RG).')
param vnetResourceGroup string = 'MC_gpu-benchmark_multi-gpu-benchmark_centralus'

@description('New subnet to create in the vnet for the Container Apps Environment. Must be empty and delegated to Microsoft.App/environments.')
param containerAppSubnetName string = 'snet-containerapp'

@description('Address prefix for the new Container Apps subnet. Must lie inside the vnet (10.224.0.0/12), not overlap existing subnets, and be at least /27.')
param containerAppSubnetPrefix string = '10.226.0.0/27'

@description('If true, the Container Apps Environment ingress is reachable only from inside the vnet.')
param containerAppEnvInternal bool = false

@description('Common tags applied to every resource.')
param tags object = {
  workload: 'llm-calculator'
  managedBy: 'bicep'
}

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: resourceGroupLocation
  tags: tags
}

module resources 'resources.bicep' = {
  name: 'gpu-benchmark-resources'
  scope: rg
  params: {
    location: location
    tags: tags
    containerAppName: containerAppName
    storageAccountName: storageAccountName
    storageAccountResourceGroup: storageAccountResourceGroup
    storageContainerName: storageContainerName
    acrName: acrName
    acrResourceGroup: acrResourceGroup
    containerImage: containerImage
    entraClientId: entraClientId
    entraTenantId: entraTenantId
    vnetName: vnetName
    vnetResourceGroup: vnetResourceGroup
    containerAppSubnetName: containerAppSubnetName
    containerAppSubnetPrefix: containerAppSubnetPrefix
    containerAppEnvInternal: containerAppEnvInternal
    kubeconfigB64: kubeconfigB64
    hfToken: hfToken
    openAiAccountName: openAiAccountName
    openAiResourceGroup: openAiResourceGroup
    aksClusterName: aksClusterName
    aksResourceGroup: aksResourceGroup
  }
}

output containerAppFqdn string = resources.outputs.containerAppFqdn
output containerAppPrincipalId string = resources.outputs.containerAppPrincipalId
output acrLoginServer string = resources.outputs.acrLoginServer
