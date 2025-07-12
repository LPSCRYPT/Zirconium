
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing Blockchain Verification...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Load proof data
    const proofData = JSON.parse(fs.readFileSync("proof.json", "utf8"));
    console.log("🔐 Proof loaded, size:", JSON.stringify(proofData).length, "chars");
    
    // Get contract instance
    const WeatherVerifier = await hre.ethers.getContractFactory("Halo2Verifier");
    const verifier = WeatherVerifier.attach(deploymentInfo.address);
    
    // Extract public instances from proof
    const publicInstances = proofData.instances[0];
    console.log("📊 Public instances:", publicInstances.length);
    console.log("🔢 Values:", publicInstances.slice(0, 4).join(", "));
    
    // Convert proof to bytes
    const proofBytes = "0x" + Buffer.from(proofData.proof).toString("hex");
    console.log("📦 Proof bytes length:", proofBytes.length);
    
    try {
        console.log("⏳ Calling verifyProof on blockchain...");
        
        // Call verify function
        const result = await verifier.verifyProof(proofBytes, publicInstances);
        
        if (result) {
            console.log("🎉 SUCCESS: Blockchain verification passed!");
            console.log("✅ Proof verified on-chain");
        } else {
            console.log("❌ FAILED: Blockchain verification rejected");
        }
        
        return result;
        
    } catch (error) {
        console.error("💥 Blockchain verification error:", error.message);
        return false;
    }
}

main()
    .then((success) => {
        if (success) {
            console.log("🎯 Test completed successfully!");
        } else {
            console.log("💥 Test failed");
        }
        process.exit(success ? 0 : 1);
    })
    .catch((error) => {
        console.error("💥 Test error:", error);
        process.exit(1);
    });
