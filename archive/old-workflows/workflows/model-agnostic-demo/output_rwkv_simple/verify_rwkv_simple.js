
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing rwkv_simple Blockchain Verification...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("../deployment_rwkv_simple.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Read calldata
    const calldataBytes = fs.readFileSync("calldata.bytes");
    const calldata = "0x" + calldataBytes.toString("hex");
    
    console.log("📦 Calldata length:", calldata.length);
    
    try {
        const [signer] = await hre.ethers.getSigners();
        
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
        
        // Log complete transaction receipt
        console.log("\n📋 FULL TRANSACTION RECEIPT:");
        console.log("=" * 50);
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("📦 Block number:", receipt.blockNumber);
        console.log("📦 Block hash:", receipt.blockHash);
        console.log("📊 Transaction index:", receipt.index);
        console.log("👤 From:", receipt.from);
        console.log("🎯 To:", receipt.to);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("⛽ Cumulative gas used:", receipt.cumulativeGasUsed.toString());
        console.log("💰 Effective gas price:", receipt.gasPrice?.toString() || "N/A");
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS (1)" : "FAILED (0)");
        console.log("📝 Logs count:", receipt.logs.length);
        console.log("🔍 Logs:", JSON.stringify(receipt.logs, null, 2));
        console.log("=" * 50);
        
        if (receipt.status === 1) {
            console.log("🎉 rwkv_simple proof verified on blockchain!");
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
