const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🔧 Debugging GAN Blockchain Verification");
    console.log("================================================================");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment_gan_simple.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Read calldata
    const calldataBytes = fs.readFileSync("output_gan_simple/calldata.bytes");
    const calldata = "0x" + calldataBytes.toString("hex");
    
    console.log("📦 Calldata length:", calldata.length);
    console.log("📏 Calldata first 100 chars:", calldata.substring(0, 100));
    
    try {
        const [signer] = await hre.ethers.getSigners();
        console.log("👤 Using signer:", signer.address);
        
        // Method 1: Low-level transaction call (proven working approach)
        console.log("\n🔧 Method 1: Low-level transaction call (like simple_mlp)");
        
        const gasEstimate = await signer.estimateGas({
            to: deploymentInfo.address,
            data: calldata
        });
        console.log("⛽ Gas estimate:", gasEstimate.toString());
        
        const tx = await signer.sendTransaction({
            to: deploymentInfo.address,
            data: calldata,
            gasLimit: gasEstimate
        });
        
        const receipt = await tx.wait();
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        if (receipt.status === 1) {
            console.log("🎉 GAN proof verified on blockchain using low-level call!");
            
            // Save results
            const results = {
                model: "gan_simple",
                status: "PROVEN", 
                gasUsed: receipt.gasUsed.toString(),
                transactionHash: receipt.hash,
                verifierAddress: deploymentInfo.address,
                method: "low_level_transaction",
                description: "Ultra-simple GAN Generator for 8x8 image generation",
                architecture: "3-layer MLP Generator",
                domain: "generative_modeling",
                timestamp: new Date().toISOString()
            };
            
            fs.writeFileSync("gan_verification_success.json", JSON.stringify(results, null, 2));
            console.log("📁 Results saved to gan_verification_success.json");
            
            return "PROVEN";
        } else {
            console.log("💥 Low-level verification failed");
            return "FAILED";
        }
        
    } catch (error) {
        console.error("💥 Error during verification:", error.message);
        
        // Try to get more details about the error
        if (error.data) {
            console.error("🔍 Error data:", error.data);
        }
        if (error.reason) {
            console.error("🔍 Error reason:", error.reason);
        }
        
        return "ERROR";
    }
}

main()
    .then((result) => {
        console.log("\n✅ GAN verification debug completed with result:", result);
        process.exit(result === "PROVEN" ? 0 : 1);
    })
    .catch((error) => {
        console.error("💥 Debug failed:", error);
        process.exit(1);
    });