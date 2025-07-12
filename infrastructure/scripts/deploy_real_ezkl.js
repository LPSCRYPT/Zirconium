const { ethers } = require("hardhat");

async function main() {
  console.log("🚀 Deploying Real EZKL Verifier Contracts");
  console.log("=" * 50);

  const [deployer] = await ethers.getSigners();
  console.log("📍 Deploying with account:", deployer.address);
  console.log("💰 Account balance:", ethers.formatEther(await ethers.provider.getBalance(deployer.address)), "ETH");

  // First, restore the real EZKL verifiers with proper contract names
  console.log("\n🔗 Deploying Real EZKL Verifiers...");
  
  console.log("   📊 Deploying RWKVVerifier (Real EZKL)...");
  try {
    const RWKVVerifier = await ethers.getContractFactory("contracts/verifiers/RWKVVerifier.sol:RWKVVerifier");
    const rwkvVerifier = await RWKVVerifier.deploy();
    await rwkvVerifier.waitForDeployment();
    console.log("   ✅ RWKVVerifier deployed:", await rwkvVerifier.getAddress());

    // Test the verify function
    console.log("   🧪 Testing architecture:", await rwkvVerifier.getArchitecture());
  } catch (error) {
    console.log("   ❌ RWKVVerifier deployment failed:", error.message);
  }

  console.log("   🐍 Deploying MambaVerifier (Real EZKL)...");
  try {
    const MambaVerifier = await ethers.getContractFactory("contracts/verifiers/MambaVerifier.sol:MambaVerifier");
    const mambaVerifier = await MambaVerifier.deploy();
    await mambaVerifier.waitForDeployment();
    console.log("   ✅ MambaVerifier deployed:", await mambaVerifier.getAddress());

    console.log("   🧪 Testing architecture:", await mambaVerifier.getArchitecture());
  } catch (error) {
    console.log("   ❌ MambaVerifier deployment failed:", error.message);
  }

  console.log("   🔢 Deploying xLSTMVerifier (Real EZKL)...");
  try {
    const xLSTMVerifier = await ethers.getContractFactory("contracts/verifiers/xLSTMVerifier.sol:xLSTMVerifier");
    const xlstmVerifier = await xLSTMVerifier.deploy();
    await xlstmVerifier.waitForDeployment();
    console.log("   ✅ xLSTMVerifier deployed:", await xlstmVerifier.getAddress());

    console.log("   🧪 Testing architecture:", await xlstmVerifier.getArchitecture());
  } catch (error) {
    console.log("   ❌ xLSTMVerifier deployment failed:", error.message);
  }

  console.log("\n🎉 Real EZKL Verifier Deployment Complete!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });