
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing gan_simple Blockchain Verification...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment_gan_simple.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Read calldata
    const calldataBytes = fs.readFileSync("output_gan_simple/calldata.bytes");
    const calldata = "0x" + calldataBytes.toString("hex");
    
    console.log("📦 Calldata length:", calldata.length);
    
    try {
        const [signer] = await hre.ethers.getSigners();
        
        // Get verifier contract instance
        const verifier = await hre.ethers.getContractAt("contracts/gan_simpleVerifier.sol:Halo2Verifier", deploymentInfo.address);
        
        const gasEstimate = await verifier.verifyProof.estimateGas(calldata);
        console.log("⛽ Gas estimate:", gasEstimate.toString());
        
        const tx = await verifier.verifyProof(calldata);
        
        const receipt = await tx.wait();
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        if (receipt.status === 1) {
            console.log("🎉 gan_simple proof verified on blockchain!");
        }
        
        return receipt.status === 1;
        
    } catch (error) {
        console.error("💥 Verification failed:", error.message);
        return false;
    }
}

main()
    .then((success) => process.exit(success ? 0 : 1))
    .catch((error) => {
        console.error("💥 Error:", error);
        process.exit(1);
    });
