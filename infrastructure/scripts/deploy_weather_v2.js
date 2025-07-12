const { ethers } = require("hardhat");
const fs = require('fs');

async function main() {
  console.log("🚀 Deploying Weather V2 Contracts");
  console.log("="*35);

  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  
  console.log(`📡 Network: ${network.name}`);
  console.log(`🔗 Chain ID: ${network.chainId}`);
  console.log(`👤 Deployer: ${deployer.address}`);
  
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`💰 Balance: ${ethers.formatEther(balance)} ETH`);
  
  // Deploy VKA first
  console.log("\n📝 Deploying WeatherVKA...");
  const WeatherVKA = await ethers.deployContract("WeatherVKA");
  await WeatherVKA.waitForDeployment();
  
  console.log("✅ WeatherVKA deployed!");
  console.log(`   📍 Address: ${await WeatherVKA.getAddress()}`);
  
  // Deploy reusable verifier
  console.log("\n📝 Deploying WeatherVerifierV2...");
  const WeatherVerifierV2 = await ethers.deployContract("WeatherVerifierV2");
  await WeatherVerifierV2.waitForDeployment();
  
  console.log("✅ WeatherVerifierV2 deployed!");
  console.log(`   📍 Address: ${await WeatherVerifierV2.getAddress()}`);

  // Save deployment info
  const deploymentInfo = {
    network: "localhost",
    chainId: Number(network.chainId),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      WeatherVKA: await WeatherVKA.getAddress(),
      WeatherVerifierV2: await WeatherVerifierV2.getAddress()
    }
  };

  fs.writeFileSync(
    'config/weather-v2-addresses.json',
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("\n🎉 WEATHER V2 CONTRACTS DEPLOYED SUCCESSFULLY!");
  console.log(`📄 Deployment info saved to: config/weather-v2-addresses.json`);
  console.log(`📍 VKA Address: ${deploymentInfo.contracts.WeatherVKA}`);
  console.log(`📍 Verifier Address: ${deploymentInfo.contracts.WeatherVerifierV2}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });