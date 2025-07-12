
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying rwkv_simple Verifier Contract...");
    
    // Get contract factory
    const Verifier = await hre.ethers.getContractFactory("contracts/rwkv_simpleVerifier.sol:Halo2Verifier");
    
    console.log("⏳ Deploying contract...");
    const verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    
    const address = await verifier.getAddress();
    console.log("✅ rwkv_simpleVerifier deployed to:", address);
    
    // Save deployment info
    const deploymentInfo = {
        address: address,
        model: "rwkv_simple",
        network: hre.network.name,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("deployment_rwkv_simple.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("📁 Deployment info saved");
    
    return address;
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    });
