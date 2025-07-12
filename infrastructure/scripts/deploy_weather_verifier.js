const { ethers } = require("hardhat");
const fs = require('fs');

async function main() {
  console.log("🚀 Deploying Weather Verifier Contract");
  console.log("="*40);

  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  
  console.log(`📡 Network: ${network.name}`);
  console.log(`🔗 Chain ID: ${network.chainId}`);
  console.log(`👤 Deployer: ${deployer.address}`);
  
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`💰 Balance: ${ethers.formatEther(balance)} ETH`);
  
  console.log("\n🏗️ Deploying Weather Verifier Contract");
  console.log("="*35);

  // Deploy the contract
  console.log("\n📝 Deploying WeatherVerifier...");
  const WeatherVerifier = await ethers.deployContract("WeatherVerifier");
  await WeatherVerifier.waitForDeployment();
  
  console.log("✅ WeatherVerifier deployed!");
  console.log(`   📍 Address: ${await WeatherVerifier.getAddress()}`);

  // Save deployment info
  const deploymentInfo = {
    network: "localhost",
    chainId: Number(network.chainId),
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contract: {
      WeatherVerifier: await WeatherVerifier.getAddress()
    }
  };

  fs.writeFileSync(
    'config/weather-verifier-address.json',
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("\n🎉 WEATHER VERIFIER DEPLOYED SUCCESSFULLY!");
  console.log(`📄 Deployment info saved to: config/weather-verifier-address.json`);
  console.log(`📍 Contract Address: ${deploymentInfo.contract.WeatherVerifier}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });