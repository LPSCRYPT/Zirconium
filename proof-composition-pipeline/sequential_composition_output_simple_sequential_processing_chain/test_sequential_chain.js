const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing Sequential Proof Composition Chain");
    console.log("================================================================");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("sequential_deployment.json", "utf8"));
    console.log("📍 Sequential executor:", deploymentInfo.sequentialChainExecutor);
    
    // Get contract instance
    const sequentialExecutor = await hre.ethers.getContractAt(
        "SequentialChainExecutor", 
        deploymentInfo.sequentialChainExecutor
    );
    
    // Prepare test inputs
    console.log("\n🎯 Preparing test data...");
    const initialInput = [/* Initial test input values */];
    const proofData = [
        /* Load actual proof data for each step */
    ];
    
    console.log("Initial input size:", initialInput.length);
    console.log("Proof data count:", proofData.length);
    
    try {
        console.log("\n⚡ Executing sequential chain...");
        const tx = await sequentialExecutor.executeSequentialChain(initialInput, proofData);
        const receipt = await tx.wait();
        
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        // Parse events
        const events = receipt.logs;
        console.log("\n📊 Execution Events:");
        
        for (const event of events) {
            try {
                const parsedEvent = sequentialExecutor.interface.parseLog(event);
                if (parsedEvent.name === "StepExecuted") {
                    console.log(`   Step ${parsedEvent.args.stepIndex}: ${parsedEvent.args.stepName} - ${parsedEvent.args.success ? "✅" : "❌"}`);
                    if (parsedEvent.args.success) {
                        console.log(`      Output: [${parsedEvent.args.output.join(", ")}]`);
                        console.log(`      Gas: ${parsedEvent.args.gasUsed}`);
                    }
                } else if (parsedEvent.name === "ChainCompleted") {
                    console.log(`\n🎉 Chain completed: ${parsedEvent.args.success ? "✅ SUCCESS" : "❌ FAILED"}`);
                    console.log(`   Final output: [${parsedEvent.args.finalOutput.join(", ")}]`);
                    console.log(`   Total gas: ${parsedEvent.args.totalGasUsed}`);
                }
            } catch (e) {
                // Skip unparseable events
            }
        }
        
    } catch (error) {
        console.error("💥 Sequential execution failed:", error.message);
        console.error("🔍 Error details:", error);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Test failed:", error);
        process.exit(1);
    });