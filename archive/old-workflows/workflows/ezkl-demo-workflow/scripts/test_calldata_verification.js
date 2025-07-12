const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing Blockchain Verification with EZKL Calldata...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Read the EZKL-generated calldata
    const calldataBytes = fs.readFileSync("workflows/ezkl-demo-workflow/proofs/calldata.bytes");
    const calldata = "0x" + calldataBytes.toString("hex");
    console.log("📦 Calldata length:", calldata.length);
    console.log("🔐 Calldata preview:", calldata.substring(0, 100) + "...");
    
    // Get contract instance
    const WeatherVerifier = await hre.ethers.getContractFactory("contracts/WeatherVerifier.sol:Halo2Verifier");
    const verifier = WeatherVerifier.attach(deploymentInfo.address);
    
    try {
        console.log("⏳ Calling verifier contract with EZKL calldata...");
        
        // Call the contract directly with the calldata
        const [signer] = await hre.ethers.getSigners();
        
        // Estimate gas first
        const gasEstimate = await signer.estimateGas({
            to: deploymentInfo.address,
            data: calldata
        });
        console.log("⛽ Gas estimate:", gasEstimate.toString());
        
        // Send the transaction
        const tx = await signer.sendTransaction({
            to: deploymentInfo.address,
            data: calldata,
            gasLimit: gasEstimate
        });
        
        const receipt = await tx.wait();
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Transaction status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        // Log full receipt details
        console.log("\n📋 FULL TRANSACTION RECEIPT:");
        console.log(JSON.stringify(receipt, null, 2));
        
        if (receipt.status === 1) {
            console.log("\n🎉 SUCCESS: Blockchain verification completed!");
            console.log("✅ Proof verified on-chain with EZKL calldata");
            
            // Try to decode the return value if possible
            console.log("📊 Verification Result: PROOF ACCEPTED BY BLOCKCHAIN");
            
        } else {
            console.log("❌ FAILED: Transaction failed");
        }
        
        return receipt.status === 1;
        
    } catch (error) {
        console.error("💥 Blockchain verification error:", error.message);
        return false;
    }
}

main()
    .then((success) => {
        if (success) {
            console.log("🎯 EZKL Demo Applied Successfully!");
            console.log("🔗 Weather model proof verified on blockchain!");
            console.log("✨ Following EZKL demo pattern with your model works!");
        } else {
            console.log("💥 Blockchain verification failed");
        }
        process.exit(success ? 0 : 1);
    })
    .catch((error) => {
        console.error("💥 Test error:", error);
        process.exit(1);
    });