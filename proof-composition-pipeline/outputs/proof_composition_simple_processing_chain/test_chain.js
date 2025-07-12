const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing Proof Composition Chain: simple_processing_chain");
    console.log("=" * 60);
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("chain_deployment.json", "utf8"));
    console.log("📍 Chain verifier:", deploymentInfo.chainVerifier);
    
    // Get contract instance
    const chainVerifier = await hre.ethers.getContractAt("ChainVerifier", deploymentInfo.chainVerifier);
    
    // Prepare proof steps
    const proofSteps = [];

    // Step 00: feature_extractor
    const step00Calldata = fs.readFileSync("step00_calldata.bytes");
    proofSteps.push({
        verifierContract: deploymentInfo.individualVerifiers[0],
        calldata: "0x" + step00Calldata.toString("hex"),
        modelName: "feature_extractor",
        position: 0
    });

    // Step 01: classifier
    const step01Calldata = fs.readFileSync("step01_calldata.bytes");
    proofSteps.push({
        verifierContract: deploymentInfo.individualVerifiers[1],
        calldata: "0x" + step01Calldata.toString("hex"),
        modelName: "classifier",
        position: 1
    });

    // Step 02: decision_maker
    const step02Calldata = fs.readFileSync("step02_calldata.bytes");
    proofSteps.push({
        verifierContract: deploymentInfo.individualVerifiers[2],
        calldata: "0x" + step02Calldata.toString("hex"),
        modelName: "decision_maker",
        position: 2
    });

    console.log(`📦 Prepared ${proofSteps.length} proof steps`);
    
    try {
        // Execute chain verification
        const tx = await chainVerifier.verifyChain(proofSteps);
        const receipt = await tx.wait();
        
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        if (receipt.status === 1) {
            console.log("🎉 Chain verification successful!");
        }
        
    } catch (error) {
        console.error("💥 Chain verification failed:", error.message);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Test failed:", error);
        process.exit(1);
    });
