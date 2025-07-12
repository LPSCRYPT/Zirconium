const hre = require("hardhat");
const fs = require("fs");

async function main() {
  console.log("🚀 Starting Zircuit Garfield Testnet Deployment");
  console.log("===============================================");
  
  // Get network info
  const network = hre.network.name;
  const chainId = hre.network.config.chainId;
  
  console.log(`📡 Network: ${network}`);
  console.log(`🔗 Chain ID: ${chainId}`);
  
  if (network !== "zircuit_testnet") {
    console.error("❌ This script is designed for Zircuit Garfield testnet only!");
    console.log("💡 Use: npx hardhat run scripts/deploy_zircuit_testnet.js --network zircuit_testnet");
    process.exit(1);
  }
  
  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();
  const deployerAddress = await deployer.getAddress();
  const balance = await hre.ethers.provider.getBalance(deployerAddress);
  
  console.log(`👤 Deployer: ${deployerAddress}`);
  console.log(`💰 Balance: ${hre.ethers.formatEther(balance)} ETH`);
  
  // Check minimum balance (0.01 ETH recommended)
  const minBalance = hre.ethers.parseEther("0.01");
  if (balance < minBalance) {
    console.error("❌ Insufficient balance! Need at least 0.01 ETH for deployment.");
    console.log("💡 Get Zircuit testnet ETH from bridge or faucet");
    console.log("🔗 Bridge: https://bridge.garfield-testnet.zircuit.com/");
    process.exit(1);
  }
  
  const deploymentResults = {
    timestamp: new Date().toISOString(),
    network: network,
    chainId: chainId,
    deployer: deployerAddress,
    initialBalance: hre.ethers.formatEther(balance),
    contracts: {},
    totalGasUsed: 0,
    totalCostETH: "0",
    deploymentHashes: []
  };
  
  const contracts = [
    "contracts/verifiers/ProductionRWKVVerifier.sol:ProductionRWKVVerifier",
    "contracts/verifiers/ProductionMambaVerifier.sol:ProductionMambaVerifier", 
    "contracts/verifiers/ProductionxLSTMVerifier.sol:ProductionxLSTMVerifier"
  ];
  
  console.log("\n🏗️ Deploying Contracts");
  console.log("======================");
  
  for (const contractPath of contracts) {
    const contractName = contractPath.split(':')[1];
    console.log(`\n📝 Deploying ${contractName}...`);
    
    try {
      // Get contract factory
      const Contract = await hre.ethers.getContractFactory(contractPath);
      
      // Deploy with gas limit  
      const contract = await Contract.deploy({
        gasLimit: 2500000 // 2.5M gas limit for Zircuit
      });
      
      // Wait for deployment
      await contract.waitForDeployment();
      const deployedAddress = await contract.getAddress();
      
      // Get deployment transaction
      const deployTx = contract.deploymentTransaction();
      const receipt = await deployTx.wait();
      
      const gasUsed = receipt.gasUsed;
      const gasPrice = deployTx.gasPrice;
      const cost = gasUsed * gasPrice;
      
      console.log(`✅ ${contractName} deployed!`);
      console.log(`   📍 Address: ${deployedAddress}`);
      console.log(`   ⛽ Gas used: ${gasUsed.toString()}`);
      console.log(`   💸 Cost: ${hre.ethers.formatEther(cost)} ETH`);
      console.log(`   🔗 Tx hash: ${deployTx.hash}`);
      
      // Store results
      deploymentResults.contracts[contractName] = {
        address: deployedAddress,
        gasUsed: gasUsed.toString(),
        costETH: hre.ethers.formatEther(cost),
        txHash: deployTx.hash,
        blockNumber: receipt.blockNumber
      };
      
      deploymentResults.totalGasUsed += Number(gasUsed);
      deploymentResults.deploymentHashes.push(deployTx.hash);
      
      // Brief pause between deployments
      await new Promise(resolve => setTimeout(resolve, 2000));
      
    } catch (error) {
      console.error(`❌ Failed to deploy ${contractName}:`, error.message);
      deploymentResults.contracts[contractName] = {
        error: error.message,
        failed: true
      };
    }
  }
  
  // Calculate total cost
  const finalBalance = await hre.ethers.provider.getBalance(deployerAddress);
  const totalCost = balance - finalBalance;
  deploymentResults.totalCostETH = hre.ethers.formatEther(totalCost);
  deploymentResults.finalBalance = hre.ethers.formatEther(finalBalance);
  
  console.log("\n📊 Deployment Summary");
  console.log("====================");
  console.log(`⛽ Total Gas Used: ${deploymentResults.totalGasUsed.toLocaleString()}`);
  console.log(`💸 Total Cost: ${deploymentResults.totalCostETH} ETH`);
  console.log(`💰 Remaining Balance: ${deploymentResults.finalBalance} ETH`);
  
  // Count successful deployments
  const successfulDeployments = Object.values(deploymentResults.contracts)
    .filter(contract => !contract.failed).length;
  
  console.log(`✅ Successful Deployments: ${successfulDeployments}/${contracts.length}`);
  
  if (successfulDeployments === contracts.length) {
    console.log("\n🎉 ALL CONTRACTS DEPLOYED SUCCESSFULLY!");
    
    // Display contract addresses for easy access
    console.log("\n📍 Contract Addresses:");
    console.log("=====================");
    Object.entries(deploymentResults.contracts).forEach(([name, data]) => {
      if (!data.failed) {
        console.log(`${name}: ${data.address}`);
      }
    });
    
    // Now deploy the orchestrator with the verifier addresses
    console.log("\n🎭 Deploying AgenticOrchestrator...");
    try {
      const Orchestrator = await hre.ethers.getContractFactory("AgenticOrchestrator");
      const orchestrator = await Orchestrator.deploy(
        deploymentResults.contracts.ProductionRWKVVerifier.address,
        deploymentResults.contracts.ProductionMambaVerifier.address,
        deploymentResults.contracts.ProductionxLSTMVerifier.address,
        {
          gasLimit: 3000000
        }
      );
      
      await orchestrator.waitForDeployment();
      const orchestratorAddress = await orchestrator.getAddress();
      
      const deployTx = orchestrator.deploymentTransaction();
      const receipt = await deployTx.wait();
      
      const gasUsed = receipt.gasUsed;
      const gasPrice = deployTx.gasPrice;
      const cost = gasUsed * gasPrice;
      
      console.log(`✅ AgenticOrchestrator deployed!`);
      console.log(`   📍 Address: ${orchestratorAddress}`);
      console.log(`   ⛽ Gas used: ${gasUsed.toString()}`);
      console.log(`   💸 Cost: ${hre.ethers.formatEther(cost)} ETH`);
      console.log(`   🔗 Tx hash: ${deployTx.hash}`);
      
      deploymentResults.contracts.AgenticOrchestrator = {
        address: orchestratorAddress,
        gasUsed: gasUsed.toString(),
        costETH: hre.ethers.formatEther(cost),
        txHash: deployTx.hash,
        blockNumber: receipt.blockNumber
      };
      
    } catch (error) {
      console.error(`❌ Failed to deploy AgenticOrchestrator:`, error.message);
      deploymentResults.contracts.AgenticOrchestrator = {
        error: error.message,
        failed: true
      };
    }
    
    // Zircuit Explorer links
    console.log("\n🔍 Zircuit Garfield Testnet Explorer:");
    console.log("===================================");
    Object.entries(deploymentResults.contracts).forEach(([name, data]) => {
      if (!data.failed) {
        console.log(`${name}: https://explorer.garfield-testnet.zircuit.com/address/${data.address}`);
      }
    });
    
  } else {
    console.log("\n⚠️ Some deployments failed. Check the results above.");
  }
  
  // Save deployment results
  const resultsFile = `zircuit-testnet-deployment-${Date.now()}.json`;
  fs.writeFileSync(resultsFile, JSON.stringify(deploymentResults, null, 2));
  console.log(`\n📄 Results saved to: ${resultsFile}`);
  
  // Create a simple addresses file for easy reference
  const successfulContracts = Object.fromEntries(
    Object.entries(deploymentResults.contracts)
      .filter(([_, data]) => !data.failed)
      .map(([name, data]) => [name, data.address])
  );
  
  if (Object.keys(successfulContracts).length > 0) {
    fs.writeFileSync('config/addresses/zircuit-testnet-addresses.json', JSON.stringify({
      network: "zircuit_testnet",
      chainId: chainId,
      timestamp: deploymentResults.timestamp,
      contracts: successfulContracts
    }, null, 2));
    console.log("📍 Addresses saved to: config/addresses/zircuit-testnet-addresses.json");
  }
  
  console.log("\n🎯 Next Steps:");
  console.log("1. View deployed contracts on Zircuit Garfield Explorer");
  console.log("2. Test the deployed contracts with zircuit_testnet_test.js");
  console.log("3. Add to MetaMask: https://garfield-testnet.zircuit.com (Chain ID: 48898)");
  console.log("4. Update documentation with the new addresses");
  console.log("5. Test ZK proof verification on Zircuit's enhanced ZK infrastructure");
  
  return deploymentResults;
}

main()
  .then((results) => {
    const successCount = Object.values(results.contracts).filter(c => !c.failed).length;
    process.exit(successCount >= 3 ? 0 : 1);
  })
  .catch((error) => {
    console.error("💥 Deployment script failed:", error);
    process.exit(1);
  });