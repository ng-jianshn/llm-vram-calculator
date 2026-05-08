// Grants the supplied principal the AKS roles needed to manage workloads
// in a single cluster:
//   - Azure Kubernetes Service Cluster User Role  (kubeconfig issuance)
//   - Azure Kubernetes Service RBAC Writer        (CRUD on K8s objects via Azure RBAC)
// Both are scoped to the cluster.
//
// Deploy this module scoped to the resource group that owns the AKS cluster.
param aksClusterName string
param principalId string

// Built-in role definition IDs
var clusterUserRoleId = '4abbcc35-e782-43d8-92c5-2d3f1bd2253f'
var rbacWriterRoleId = 'a7ffa36f-339b-4b5c-8bdf-e2c188b2c0eb'

resource aks 'Microsoft.ContainerService/managedClusters@2024-05-01' existing = {
  name: aksClusterName
}

resource clusterUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: aks
  name: guid(aks.id, principalId, clusterUserRoleId)
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', clusterUserRoleId)
  }
}

resource rbacWriter 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: aks
  name: guid(aks.id, principalId, rbacWriterRoleId)
  properties: {
    principalId: principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', rbacWriterRoleId)
  }
}
