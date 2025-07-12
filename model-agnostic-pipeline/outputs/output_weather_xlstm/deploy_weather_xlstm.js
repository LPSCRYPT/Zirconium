
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying weather_xlstm Verifier Contract...");
    
    // Get contract factory
    const Verifier = await hre.ethers.getContractFactory("contracts/weather_xlstmVerifier.sol:Halo2Verifier");
    
    console.log("⏳ Deploying contract...");
    const verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    
    const address = await verifier.getAddress();
    console.log("✅ weather_xlstmVerifier deployed to:", address);
    
    // Save deployment info
    const deploymentInfo = {
        address: address,
        model: "weather_xlstm",
        network: hre.network.name,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("deployment_weather_xlstm.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("📁 Deployment info saved");
    
    return address;
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    });
