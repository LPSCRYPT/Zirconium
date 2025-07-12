const { ethers } = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🔧 Debug GAN Compact Blockchain Verification...");
    console.log("Using low-level transaction approach (same as working models)");
    
    // Get signer
    const [signer] = await ethers.getSigners();
    console.log("📍 Signer address:", signer.address);
    
    // Load deployment info
    let deploymentInfo;
    try {
        const deploymentData = fs.readFileSync('output_gan_compact/deployment_gan_compact.json', 'utf8');
        deploymentInfo = JSON.parse(deploymentData);
        console.log("📍 Contract address:", deploymentInfo.address);
    } catch (error) {
        console.error("❌ Error loading deployment info:", error.message);
        return;
    }
    
    // Load calldata as hex
    let calldata;
    try {
        const calldataBuffer = fs.readFileSync('output_gan_compact/calldata.bytes');
        calldata = '0x' + calldataBuffer.toString('hex');
        console.log("📦 Calldata length:", calldata.length);
        console.log("📦 Calldata preview:", calldata.substring(0, 100) + "...");
    } catch (error) {
        console.error("❌ Error loading calldata:", error.message);
        return;
    }
    
    try {
        console.log("\n🚀 Attempting verification with low-level transaction...");
        
        // Estimate gas first
        const gasEstimate = await signer.estimateGas({
            to: deploymentInfo.address,
            data: calldata
        });
        console.log("⛽ Gas estimate:", gasEstimate.toString());
        
        // Send transaction with increased gas limit
        const gasLimit = Math.floor(Number(gasEstimate) * 1.2); // 20% buffer
        
        const tx = await signer.sendTransaction({
            to: deploymentInfo.address,
            data: calldata,
            gasLimit: gasLimit
        });
        
        console.log("📨 Transaction sent:", tx.hash);
        console.log("⏳ Waiting for confirmation...");
        
        const receipt = await tx.wait();
        
        if (receipt.status === 1) {
            console.log("\n🎉 GAN COMPACT BLOCKCHAIN VERIFICATION SUCCESSFUL!");
            console.log("================================================================");
            console.log("📋 Transaction hash:", receipt.transactionHash);
            console.log("⛽ Gas used:", receipt.gasUsed.toString());
            console.log("🎨 Model: Compact GAN (16x16 RGB)");
            console.log("📊 Circuit rows: 174,641");
            console.log("💰 Verification cost:", receipt.gasUsed.toString(), "gas");
            console.log("✅ Status: SUCCESS");
            console.log("💡 First 16x16 RGB GAN verified on blockchain!");
            
            // Save verification result
            const verificationResult = {
                success: true,
                transactionHash: receipt.transactionHash,
                gasUsed: receipt.gasUsed.toString(),
                contractAddress: deploymentInfo.address,
                model: "gan_compact",
                resolution: "16x16 RGB",
                circuitRows: 174641,
                timestamp: new Date().toISOString()
            };
            
            fs.writeFileSync(
                'gan_compact_verification_success.json', 
                JSON.stringify(verificationResult, null, 2)
            );
            
        } else {
            console.log("❌ Transaction failed");
            console.log("Receipt:", receipt);
        }
        
    } catch (error) {
        console.error("💥 Verification failed:", error.message);
        
        if (error.message.includes("revert")) {
            console.log("💡 This might be a proof validation error");
            console.log("💡 Check that the proof and calldata are correctly generated");
        }
        
        if (error.message.includes("gas")) {
            console.log("💡 This might be a gas limit issue");
            console.log("💡 Try increasing gas limit or deploying on L2");
        }
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });