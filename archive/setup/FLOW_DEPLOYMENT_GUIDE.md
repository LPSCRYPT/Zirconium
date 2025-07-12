# Flow EVM Deployment Guide

This guide explains how to deploy your zirconium contracts to Flow EVM testnet and mainnet.

## Overview

Flow EVM allows you to deploy Solidity contracts without any code changes, using familiar tools like Hardhat and MetaMask. Your existing zirconium contracts are fully compatible.

## Prerequisites

1. **Node.js and npm** installed
2. **Flow testnet tokens** for deployment
3. **MetaMask** configured for Flow EVM
4. **Private key** with sufficient FLOW balance

## Network Configuration

The Hardhat configuration has been updated with Flow networks:

### Flow EVM Testnet
- **Network Name**: flow_testnet
- **RPC URL**: https://testnet.evm.nodes.onflow.org
- **Chain ID**: 545
- **Currency**: FLOW
- **Explorer**: https://evm-testnet.flowscan.io/

### Flow EVM Mainnet
- **Network Name**: flow_mainnet
- **RPC URL**: https://mainnet.evm.nodes.onflow.org
- **Chain ID**: 747
- **Currency**: FLOW
- **Explorer**: https://evm.flowscan.io/

## Environment Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment variables** in `.env`:
   ```bash
   PRIVATE_KEY=your_private_key_here
   FLOW_TESTNET_URL=https://testnet.evm.nodes.onflow.org
   FLOW_MAINNET_URL=https://mainnet.evm.nodes.onflow.org
   ```

3. **Get testnet tokens**:
   Visit https://testnet-faucet.onflow.org/ to get FLOW tokens for testing

## MetaMask Configuration

### Add Flow EVM Testnet to MetaMask

1. Open MetaMask
2. Click network dropdown → "Add Network"
3. Enter the following details:
   - **Network Name**: Flow EVM Testnet
   - **New RPC URL**: https://testnet.evm.nodes.onflow.org
   - **Chain ID**: 545
   - **Currency Symbol**: FLOW
   - **Block Explorer**: https://evm-testnet.flowscan.io/

### Add Flow EVM Mainnet to MetaMask

1. Open MetaMask
2. Click network dropdown → "Add Network"
3. Enter the following details:
   - **Network Name**: Flow EVM Mainnet
   - **New RPC URL**: https://mainnet.evm.nodes.onflow.org
   - **Chain ID**: 747
   - **Currency Symbol**: FLOW
   - **Block Explorer**: https://evm.flowscan.io/

## Deployment Commands

### Deploy to Flow EVM Testnet

```bash
npx hardhat run scripts/deploy_flow_testnet.js --network flow_testnet
```

This will deploy:
- ProductionRWKVVerifier
- ProductionMambaVerifier
- ProductionxLSTMVerifier
- AgenticOrchestrator (with verifier addresses)

### Deploy to Flow EVM Mainnet

```bash
npx hardhat run scripts/deploy_flow_mainnet.js --network flow_mainnet
```

⚠️ **Warning**: Mainnet deployment costs real FLOW tokens. Test thoroughly on testnet first!

## What the Deployment Scripts Do

1. **Verify network and balance** - Ensures you're on the correct network with sufficient funds
2. **Deploy verifier contracts** - Deploys the three neural architecture verifiers
3. **Deploy orchestrator** - Deploys the main coordination contract with verifier addresses
4. **Generate reports** - Creates JSON files with deployment results and addresses
5. **Display explorer links** - Shows URLs to view contracts on Flow EVM Explorer

## After Deployment

The scripts will generate:
- `config/addresses/flow-testnet-addresses.json` - Contract addresses for testnet
- `config/addresses/flow-mainnet-addresses.json` - Contract addresses for mainnet
- Detailed deployment logs with gas usage and costs

## Testing Deployed Contracts

After deployment, you can:

1. **View on Explorer**: Check contract deployment on Flow EVM Explorer
2. **Interact via Hardhat**: Use Hardhat console to interact with contracts
3. **Test functionality**: Run your existing test suites against deployed contracts

## Flow EVM Advantages

- **No code changes required** - Your Solidity contracts work as-is
- **Lower gas fees** - More cost-effective than Ethereum
- **Fast transactions** - Better performance than Ethereum
- **MEV resistance** - Built-in protection against MEV attacks
- **Native Flow features** - Access to Flow's unique capabilities

## Troubleshooting

### Common Issues

1. **Insufficient balance**: Get more FLOW tokens from the faucet
2. **Gas estimation failed**: Increase gas limit in deployment scripts
3. **Network connection**: Check RPC endpoint connectivity
4. **Private key issues**: Ensure private key is correctly formatted

### Support Resources

- [Flow Developer Portal](https://developers.flow.com/)
- [Flow EVM Documentation](https://developers.flow.com/evm/)
- [Flow Discord](https://discord.gg/flow)

## Security Considerations

1. **Never commit private keys** to git repositories
2. **Use environment variables** for sensitive data
3. **Test thoroughly** on testnet before mainnet deployment
4. **Verify contracts** on the explorer when possible
5. **Monitor deployed contracts** for unexpected behavior

## Cost Estimation

Deployment costs depend on:
- Contract size and complexity
- Current network gas prices
- Number of contracts being deployed

Typical costs:
- **Testnet**: Free (using faucet tokens)
- **Mainnet**: ~0.01-0.1 FLOW per contract (varies by complexity)

## Next Steps

1. Deploy to testnet and verify functionality
2. Test all contract interactions
3. Deploy to mainnet when ready
4. Update documentation with deployed addresses
5. Monitor contracts using Flow EVM Explorer