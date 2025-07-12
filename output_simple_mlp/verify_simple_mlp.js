
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing simple_mlp Blockchain Verification...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment_simple_mlp.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Load proof data
    const proofData = JSON.parse(fs.readFileSync("proof.json", "utf8"));
    console.log("🔐 Proof loaded, size:", JSON.stringify(proofData).length, "chars");
    
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
        const [signer] = await hre.ethers.getSigners();
        
        // Get verifier contract instance
        const verifier = await hre.ethers.getContractAt("contracts/simple_mlpVerifier.sol:Halo2Verifier", deploymentInfo.address);
        
        console.log("⏳ Calling verifyProof on blockchain...");
        
        // Estimate gas first
        const gasEstimate = await verifier.verifyProof.estimateGas(proofBytes, publicInstances);
        console.log("⛽ Gas estimate:", gasEstimate.toString());
        
        // Call verify function
        const tx = await verifier.verifyProof(proofBytes, publicInstances);
        const receipt = await tx.wait();
        
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        // The function should return true/false
        const result = await verifier.verifyProof.staticCall(proofBytes, publicInstances);
        
        if (result) {
            console.log("🎉 SUCCESS: Blockchain verification passed!");
            console.log("✅ simple_mlp proof verified on blockchain!");
        } else {
            console.log("❌ FAILED: Blockchain verification rejected");
        }
        
        return result;
        
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
