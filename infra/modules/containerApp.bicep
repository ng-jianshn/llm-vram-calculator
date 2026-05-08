// automated-gpu-benchmark Container App: user-assigned MI, AAD-based ACR pull,
// port 5000 ingress, and Microsoft Entra (Easy Auth) configured to
// require login against the existing enterprise app.
param name string
param location string
param tags object

param environmentId string
param userAssignedIdentityId string
param userAssignedIdentityClientId string

param containerImage string
param acrLoginServer string

param storageAccountName string
param storageContainerName string

param entraClientId string
param entraTenantId string

@description('Base64-encoded kubeconfig (kubelogin MSI mode). Stored as secret kube-config and exposed as env var KUBECONFIG_B64.')
@secure()
param kubeconfigB64 string

@description('Hugging Face access token. Stored as secret hf-token and exposed as env var HF_TOKEN.')
@secure()
param hfToken string

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentityId}': {}
    }
  }
  properties: {
    environmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 5000
        transport: 'auto'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: acrLoginServer
          identity: userAssignedIdentityId
        }
      ]
      secrets: [
        {
          name: 'kube-config'
          value: kubeconfigB64
        }
        {
          name: 'hf-token'
          value: hfToken
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'app'
          image: containerImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'AZURE_CLIENT_ID'
              value: userAssignedIdentityClientId
            }
            {
              name: 'BENCHMARK_STORAGE_ACCOUNT'
              value: storageAccountName
            }
            {
              name: 'BENCHMARK_STORAGE_CONTAINER'
              value: storageContainerName
            }
            {
              name: 'BENCHMARK_STORAGE_ENDPOINT_SUFFIX'
              value: 'blob.${environment().suffixes.storage}'
            }
            {
              name: 'KUBECONFIG_B64'
              secretRef: 'kube-config'
            }
            {
              name: 'HF_TOKEN'
              secretRef: 'hf-token'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

// Microsoft Entra ID authentication (Easy Auth) – requires login.
resource auth 'Microsoft.App/containerApps/authConfigs@2024-03-01' = {
  parent: app
  name: 'current'
  properties: {
    platform: {
      enabled: true
    }
    globalValidation: {
      unauthenticatedClientAction: 'RedirectToLoginPage'
      redirectToProvider: 'azureactivedirectory'
    }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          clientId: entraClientId
          // No clientSecretSettingName: auth uses a federated credential
          // on the app registration, signed by the Easy Auth platform.
          openIdIssuer: 'https://sts.windows.net/${entraTenantId}/v2.0'
        }
        validation: {
          allowedAudiences: [
            'api://${entraClientId}'
          ]
        }
      }
    }
    login: {
      tokenStore: {
        enabled: false
      }
      preserveUrlFragmentsForLogins: false
    }
  }
}

output id string = app.id
output name string = app.name
output fqdn string = app.properties.configuration.ingress.fqdn
