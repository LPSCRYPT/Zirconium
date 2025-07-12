const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying Weather Verifier Contract...");
    
    // Get contract factory
    const WeatherVerifier = await hre.ethers.getContractFactory("contracts/WeatherVerifier.sol:Halo2Verifier");
    
    console.log("⏳ Deploying contract...");
    const verifier = await WeatherVerifier.deploy();
    await verifier.waitForDeployment();
    
    const address = await verifier.getAddress();
    console.log("✅ WeatherVerifier deployed to:", address);
    
    // Save deployment info
    const deploymentInfo = {
        address: address,
        network: hre.network.name,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("deployment.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("📁 Deployment info saved to deployment.json");
    
    return address;
}

main()
    .then((address) => {
        console.log("🎉 Deployment successful!");
        process.exit(0);
    })
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    });