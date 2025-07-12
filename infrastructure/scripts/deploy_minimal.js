const { ethers } = require("hardhat");

async function main() {
  console.log("🚀 Deploying Minimal EZKL Proof-Chaining Contracts");
  console.log("=" * 50);

  const [deployer] = await ethers.getSigners();
  console.log("📍 Deploying with account:", deployer.address);
  console.log("💰 Account balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)), "ETH");

  // Deploy verifiers
  console.log("\n🔗 Deploying Verifiers...");
  
  console.log("   📊 Deploying Real EZKL RWKVVerifier...");
  const RWKVVerifier = await ethers.getContractFactory("RWKVVerifierWrapper");
  const rwkvVerifier = await RWKVVerifier.deploy();
  await rwkvVerifier.waitForDeployment();
  console.log("   ✅ RWKVVerifier deployed:", await rwkvVerifier.getAddress());
  console.log("   🔗 Underlying EZKL verifier:", await rwkvVerifier.getEZKLVerifier());

  console.log("   🐍 Deploying Real EZKL MambaVerifier...");
  const MambaVerifier = await ethers.getContractFactory("MambaVerifierWrapper");
  const mambaVerifier = await MambaVerifier.deploy();
  await mambaVerifier.waitForDeployment();
  console.log("   ✅ MambaVerifier deployed:", await mambaVerifier.getAddress());
  console.log("   🔗 Underlying EZKL verifier:", await mambaVerifier.getEZKLVerifier());

  console.log("   🔢 Deploying Real EZKL xLSTMVerifier...");
  const xLSTMVerifier = await ethers.getContractFactory("xLSTMVerifierWrapper");
  const xlstmVerifier = await xLSTMVerifier.deploy();
  await xlstmVerifier.waitForDeployment();
  console.log("   ✅ xLSTMVerifier deployed:", await xlstmVerifier.getAddress());
  console.log("   🔗 Underlying EZKL verifier:", await xlstmVerifier.getEZKLVerifier());

  // Deploy orchestrator
  console.log("\n🎭 Deploying ProofChainOrchestrator...");
  const ProofChainOrchestrator = await ethers.getContractFactory("ProofChainOrchestrator");
  const orchestrator = await ProofChainOrchestrator.deploy();
  await orchestrator.waitForDeployment();
  console.log("   ✅ ProofChainOrchestrator deployed:", await orchestrator.getAddress());

  // Test basic functionality
  console.log("\n🔍 Testing Basic Functionality...");
  
  // Test architecture names
  const rwkvArch = await rwkvVerifier.getArchitecture();
  const mambaArch = await mambaVerifier.getArchitecture();
  const xlstmArch = await xlstmVerifier.getArchitecture();
  
  console.log("   📊 RWKV Architecture:", rwkvArch);
  console.log("   🐍 Mamba Architecture:", mambaArch);
  console.log("   🔢 xLSTM Architecture:", xlstmArch);

  // Test verification counts
  const rwkvCount = await rwkvVerifier.getTotalVerifications();
  const mambaCount = await mambaVerifier.getTotalVerifications();
  const xlstmCount = await xlstmVerifier.getTotalVerifications();
  
  console.log("   📈 Initial verification counts:", rwkvCount, mambaCount, xlstmCount);

  // Test orchestrator
  const totalExecutions = await orchestrator.totalExecutions();
  console.log("   🎭 Initial executions:", totalExecutions);

  // Save deployment addresses
  const deploymentData = {
    network: "localhost",
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      RWKVVerifier: await rwkvVerifier.getAddress(),
      MambaVerifier: await mambaVerifier.getAddress(),
      xLSTMVerifier: await xlstmVerifier.getAddress(),
      ProofChainOrchestrator: await orchestrator.getAddress()
    },
    architectures: {
      rwkv: rwkvArch,
      mamba: mambaArch,
      xlstm: xlstmArch
    }
  };

  // Write to file
  const fs = require('fs');
  const path = require('path');
  
  const deploymentDir = 'config/deployments';
  if (!fs.existsSync(deploymentDir)) {
    fs.mkdirSync(deploymentDir, { recursive: true });
  }
  
  const fileName = `localhost-minimal-${Date.now()}.json`;
  const filePath = path.join(deploymentDir, fileName);
  
  fs.writeFileSync(filePath, JSON.stringify(deploymentData, null, 2));
  console.log(`\n💾 Deployment data saved to: ${filePath}`);

  console.log("\n🎉 Deployment Complete!");
  console.log("=" * 50);
  console.log("📋 Deployment Summary:");
  console.log(`   🔗 RWKVVerifier: ${await rwkvVerifier.getAddress()}`);
  console.log(`   🔗 MambaVerifier: ${await mambaVerifier.getAddress()}`);
  console.log(`   🔗 xLSTMVerifier: ${await xlstmVerifier.getAddress()}`);
  console.log(`   🔗 Orchestrator: ${await orchestrator.getAddress()}`);
  console.log("\n🎯 Ready for testing proof-chaining workflow!");

  return deploymentData;
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });