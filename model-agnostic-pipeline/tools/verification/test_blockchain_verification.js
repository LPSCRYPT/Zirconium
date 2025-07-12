const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing Blockchain Verification...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Load proof data from localhost_test
    const proofData = JSON.parse(fs.readFileSync("localhost_test/proof.json", "utf8"));
    console.log("🔐 Proof loaded, size:", JSON.stringify(proofData).length, "chars");
    
    // Get contract instance
    const WeatherVerifier = await hre.ethers.getContractFactory("contracts/WeatherVerifier.sol:Halo2Verifier");
    const verifier = WeatherVerifier.attach(deploymentInfo.address);
    
    // Extract public instances from proof and convert to BigInt
    const publicInstancesHex = proofData.instances[0];
    const publicInstances = publicInstancesHex.map(hex => {
        // Convert hex string to BigInt
        return BigInt("0x" + hex);
    });
    console.log("📊 Public instances:", publicInstances.length);
    console.log("🔢 First few values (hex):", publicInstancesHex.slice(0, 4).join(", "));
    console.log("🔢 First few values (BigInt):", publicInstances.slice(0, 4).map(x => x.toString()).join(", "));
    
    // Convert proof to bytes
    const proofBytes = "0x" + Buffer.from(proofData.proof).toString("hex");
    console.log("📦 Proof bytes length:", proofBytes.length);
    
    try {
        console.log("⏳ Calling verifyProof on blockchain...");
        
        // Estimate gas first
        const gasEstimate = await verifier.verifyProof.estimateGas(proofBytes, publicInstances);
        console.log("⛽ Gas estimate:", gasEstimate.toString());
        
        // Call verify function
        const tx = await verifier.verifyProof(proofBytes, publicInstances);
        const receipt = await tx.wait();
        
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        
        // The function should return true/false
        const result = await verifier.verifyProof.staticCall(proofBytes, publicInstances);
        
        if (result) {
            console.log("🎉 SUCCESS: Blockchain verification passed!");
            console.log("✅ Proof verified on-chain");
        } else {
            console.log("❌ FAILED: Blockchain verification rejected");
        }
        
        return result;
        
    } catch (error) {
        console.error("💥 Blockchain verification error:", error.message);
        
        // Try to get more detailed error info
        if (error.data) {
            console.error("📋 Error data:", error.data);
        }
        if (error.reason) {
            console.error("📋 Error reason:", error.reason);
        }
        
        return false;
    }
}

main()
    .then((success) => {
        if (success) {
            console.log("🎯 Blockchain verification test completed successfully!");
        } else {
            console.log("💥 Blockchain verification test failed");
        }
        process.exit(success ? 0 : 1);
    })
    .catch((error) => {
        console.error("💥 Test error:", error);
        process.exit(1);
    });