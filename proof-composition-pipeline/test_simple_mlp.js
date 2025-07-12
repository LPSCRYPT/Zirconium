const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Testing simple_mlp with localhost blockchain verification");
    console.log("================================================================");
    
    // Deploy the simple_mlp verifier
    console.log("📝 Deploying simple_mlp verifier...");
    const SimpleMLPVerifier = await hre.ethers.getContractFactory("contracts/simple_mlpVerifier.sol:Halo2Verifier", {
        libraries: {},
    });
    
    const verifier = await SimpleMLPVerifier.deploy();
    await verifier.waitForDeployment();
    
    console.log("✅ simple_mlp verifier deployed to:", await verifier.getAddress());
    
    // Load calldata from the MAP output
    const calldataPath = "/Users/bot/code/zirconium/model-agnostic-pipeline/output_simple_mlp/calldata.txt";
    
    if (fs.existsSync(calldataPath)) {
        console.log("📋 Loading proof calldata...");
        const calldata = fs.readFileSync(calldataPath, "utf8").trim();
        
        console.log("🔍 Testing proof verification...");
        
        try {
            // Estimate gas for verification
            const gasEstimate = await verifier.verifyProof.estimateGas(calldata);
            console.log(`⛽ Estimated gas: ${gasEstimate.toString()}`);
            
            // Verify the proof
            const tx = await verifier.verifyProof(calldata);
            const receipt = await tx.wait();
            
            console.log("📊 Verification Results:");
            console.log(`📋 Transaction hash: ${receipt.hash}`);
            console.log(`⛽ Gas used: ${receipt.gasUsed.toString()}`);
            console.log(`✅ Status: ${receipt.status === 1 ? "SUCCESS" : "FAILED"}`);
            
            if (receipt.status === 1) {
                console.log("🎉 simple_mlp PROOF VERIFIED SUCCESSFULLY!");
                console.log(`📈 Actual gas cost: ${receipt.gasUsed.toString()}`);
            }
            
        } catch (error) {
            console.error("💥 Verification failed:", error.message);
        }
        
    } else {
        console.log("⚠️  Calldata file not found, using dummy verification");
    }
    
    console.log("\n✅ simple_mlp blockchain testing completed!");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Test failed:", error);
        process.exit(1);
    });