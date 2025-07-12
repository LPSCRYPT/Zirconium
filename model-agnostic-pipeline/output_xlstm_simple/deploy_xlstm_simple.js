
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying xlstm_simple Verifier Contract...");
    
    // Get contract factory
    const Verifier = await hre.ethers.getContractFactory("contracts/xlstm_simpleVerifier.sol:Halo2Verifier");
    
    console.log("⏳ Deploying contract...");
    const verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    
    const address = await verifier.getAddress();
    console.log("✅ xlstm_simpleVerifier deployed to:", address);
    
    // Save deployment info
    const deploymentInfo = {
        address: address,
        model: "xlstm_simple",
        network: hre.network.name,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("deployment_xlstm_simple.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("📁 Deployment info saved");
    
    return address;
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    });
