const { ethers } = require("hardhat");

async function main() {
  console.log("🚀 Starting Localhost Deployment");
  console.log("=".repeat(35));

  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  
  console.log(`📡 Network: ${network.name}`);
  console.log(`🔗 Chain ID: ${network.chainId}`);
  console.log(`👤 Deployer: ${deployer.address}`);
  
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`💰 Balance: ${ethers.formatEther(balance)} ETH`);
  
  console.log("\n🏗️ Deploying Contracts");
  console.log("=".repeat(25));

  // Deploy xLSTM Verifier
  console.log("\n📝 Deploying xLSTMVerifier...");
  const xLSTMVerifier = await ethers.deployContract("xLSTMVerifier");
  await xLSTMVerifier.waitForDeployment();
  
  console.log("✅ xLSTMVerifier deployed!");
  console.log(`   📍 Address: ${await xLSTMVerifier.getAddress()}`);
  
  // Deploy RWKV Verifier
  console.log("\n📝 Deploying RWKVVerifier...");
  const rwkvVerifier = await ethers.deployContract("RWKVVerifier");
  await rwkvVerifier.waitForDeployment();
  
  console.log("✅ RWKVVerifier deployed!");
  console.log(`   📍 Address: ${await rwkvVerifier.getAddress()}`);
  
  // Deploy Mamba Verifier
  console.log("\n📝 Deploying MambaVerifier...");
  const mambaVerifier = await ethers.deployContract("MambaVerifier");
  await mambaVerifier.waitForDeployment();
  
  console.log("✅ MambaVerifier deployed!");
  console.log(`   📍 Address: ${await mambaVerifier.getAddress()}`);

  // Save deployment info
  const deploymentInfo = {
    network: "localhost",
    chainId: Number(network.chainId),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      xLSTMVerifier: await xLSTMVerifier.getAddress(),
      RWKVVerifier: await rwkvVerifier.getAddress(),
      MambaVerifier: await mambaVerifier.getAddress()
    }
  };

  const fs = require('fs');
  fs.writeFileSync(
    'config/localhost-addresses.json',
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("\n🎉 ALL CONTRACTS DEPLOYED SUCCESSFULLY!");
  console.log(`📄 Deployment info saved to: config/localhost-addresses.json`);

  console.log("\n📍 Contract Addresses:");
  console.log(`xLSTMVerifier: ${deploymentInfo.contracts.xLSTMVerifier}`);
  console.log(`RWKVVerifier: ${deploymentInfo.contracts.RWKVVerifier}`);
  console.log(`MambaVerifier: ${deploymentInfo.contracts.MambaVerifier}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });