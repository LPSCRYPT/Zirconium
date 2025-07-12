const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing Blockchain Verification with EZKL Calldata...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Read the EZKL-generated calldata
    const calldataBytes = fs.readFileSync("localhost_test/calldata.bytes");
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
        
        if (receipt.status === 1) {
            console.log("🎉 SUCCESS: Blockchain verification completed!");
            console.log("✅ Proof verified on-chain with EZKL calldata");
            
            // Try to decode the return value
            if (receipt.logs && receipt.logs.length > 0) {
                console.log("📋 Transaction logs:", receipt.logs.length);
            }
            
        } else {
            console.log("❌ FAILED: Transaction failed");
        }
        
        return receipt.status === 1;
        
    } catch (error) {
        console.error("💥 Blockchain verification error:", error.message);
        
        // Try to get more detailed error info
        if (error.data) {
            console.error("📋 Error data:", error.data);
        }
        if (error.reason) {
            console.error("📋 Error reason:", error.reason);
        }
        if (error.error && error.error.data) {
            console.error("📋 Inner error data:", error.error.data);
        }
        
        return false;
    }
}

main()
    .then((success) => {
        if (success) {
            console.log("🎯 EZKL calldata blockchain verification test PASSED!");
            console.log("🔗 Your weather model proof verified successfully on blockchain!");
        } else {
            console.log("💥 EZKL calldata blockchain verification test FAILED");
        }
        process.exit(success ? 0 : 1);
    })
    .catch((error) => {
        console.error("💥 Test error:", error);
        process.exit(1);
    });