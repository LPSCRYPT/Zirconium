const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🎨 Testing GAN Blockchain Verification");
    console.log("================================================================");
    
    try {
        // Deploy the GAN verifier
        console.log("📝 Deploying GAN verifier...");
        const GanVerifier = await hre.ethers.getContractFactory("contracts/gan_simpleVerifier.sol:Halo2Verifier");
        
        const verifier = await GanVerifier.deploy();
        await verifier.waitForDeployment();
        
        const address = await verifier.getAddress();
        console.log("✅ GAN verifier deployed to:", address);
        
        // Load calldata from the GAN output
        const calldataPath = "output_gan_simple/calldata.bytes";
        
        if (fs.existsSync(calldataPath)) {
            console.log("📋 Loading proof calldata...");
            const calldata = "0x" + fs.readFileSync(calldataPath).toString("hex");
            console.log(`📏 Calldata size: ${calldata.length} characters`);
            
            // Test proof verification
            console.log("🔍 Testing proof verification...");
            
            // Estimate gas
            const gasEstimate = await verifier.verifyProof.estimateGas(calldata);
            console.log(`⛽ Estimated gas: ${gasEstimate.toString()}`);
            
            // Execute verification
            console.log("⚡ Executing proof verification...");
            const tx = await verifier.verifyProof(calldata);
            const receipt = await tx.wait();
            
            console.log("\\n📊 Verification Results:");
            console.log("================================================================");
            console.log("📋 Transaction hash:", receipt.hash);
            console.log("⛽ Gas used:", receipt.gasUsed.toString());
            console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
            console.log("🔗 Block number:", receipt.blockNumber);
            
            if (receipt.status === 1) {
                console.log("\\n🎉 GAN BLOCKCHAIN VERIFICATION SUCCESSFUL!");
                console.log("================================================================");
                console.log("✅ PROVEN STATUS: gan_simple can now be marked as PROVEN");
                console.log("📈 Actual gas cost:", receipt.gasUsed.toString());
                console.log("💡 First generative model verified on blockchain!");
                
                // Save results
                const results = {
                    model: "gan_simple",
                    status: "PROVEN",
                    gasUsed: receipt.gasUsed.toString(),
                    transactionHash: receipt.hash,
                    verifierAddress: address,
                    description: "Ultra-simple GAN Generator for 8x8 image generation",
                    architecture: "3-layer MLP Generator",
                    domain: "generative_modeling",
                    timestamp: new Date().toISOString()
                };
                
                fs.writeFileSync("gan_verification_results.json", JSON.stringify(results, null, 2));
                console.log("📁 Results saved to gan_verification_results.json");
                
                return "PROVEN";
            } else {
                console.log("💥 Verification failed");
                return "FAILED";
            }
            
        } else {
            console.log("⚠️  Calldata file not found at:", calldataPath);
            return "CALLDATA_NOT_FOUND";
        }
        
    } catch (error) {
        console.error("💥 Error:", error.message);
        return "ERROR";
    }
}

main()
    .then((result) => {
        console.log("\\n✅ GAN verification completed with result:", result);
        process.exit(0);
    })
    .catch((error) => {
        console.error("💥 Test failed:", error);
        process.exit(1);
    });