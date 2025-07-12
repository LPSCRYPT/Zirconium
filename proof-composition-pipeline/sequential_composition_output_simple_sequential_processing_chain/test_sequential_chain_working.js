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
    
    // Get enhanced verifier instances for testing
    const enhancedVerifiers = [];
    const contractNames = [
        "EnhancedFeature_ExtractorVerifier",
        "EnhancedClassifierVerifier", 
        "EnhancedDecision_MakerVerifier"
    ];
    
    for (let i = 0; i < deploymentInfo.enhancedVerifiers.length; i++) {
        const verifier = await hre.ethers.getContractAt(
            contractNames[i],
            deploymentInfo.enhancedVerifiers[i]
        );
        enhancedVerifiers.push(verifier);
    }
    
    console.log("\\n📋 Chain Information:");
    const chainInfo = await sequentialExecutor.getChainInfo();
    console.log(`   Steps: ${chainInfo.stepCount}`);
    console.log(`   Step Names: ${chainInfo.stepNames.join(' → ')}`);
    
    // Test individual enhanced verifiers first
    console.log("\\n🔧 Testing Individual Enhanced Verifiers...");
    
    // Prepare test data - these need to match the actual model input/output shapes
    const testInputs = [
        // Step 0: feature_extractor (10 inputs, 8 outputs)
        Array.from({length: 10}, (_, i) => Math.floor(Math.random() * 1000)),
        // Step 1: classifier (8 inputs, 5 outputs) 
        Array.from({length: 8}, (_, i) => Math.floor(Math.random() * 1000)),
        // Step 2: decision_maker (5 inputs, 3 outputs)
        Array.from({length: 5}, (_, i) => Math.floor(Math.random() * 1000))
    ];
    
    const testOutputs = [
        // Expected outputs for each step
        Array.from({length: 8}, (_, i) => Math.floor(Math.random() * 1000)),
        Array.from({length: 5}, (_, i) => Math.floor(Math.random() * 1000)),
        Array.from({length: 3}, (_, i) => Math.floor(Math.random() * 1000))
    ];
    
    // Load actual proof calldata
    const proofData = [];
    for (let i = 0; i < 3; i++) {
        const stepName = deploymentInfo.stepNames[i];
        const calldataFile = `step${i.toString().padStart(2, '0')}_${stepName}_calldata.bytes`;
        
        if (fs.existsSync(calldataFile)) {
            const calldata = fs.readFileSync(calldataFile);
            proofData.push("0x" + calldata.toString("hex"));
            console.log(`   ✅ Loaded calldata for step ${i}: ${stepName} (${calldata.length} bytes)`);
        } else {
            // Create dummy calldata for testing
            proofData.push("0x" + "00".repeat(100)); // Dummy 100 bytes
            console.log(`   ⚠️  Using dummy calldata for step ${i}: ${stepName}`);
        }
    }
    
    // Test individual verifiers with dummy verification (they'll likely fail with dummy data)
    console.log("\\n🔍 Testing Individual Verifiers (may fail with test data):");
    
    for (let i = 0; i < enhancedVerifiers.length; i++) {
        try {
            const publicInputs = [...testInputs[i], ...testOutputs[i]];
            console.log(`   Testing step ${i} with ${publicInputs.length} public inputs...`);
            
            // Try the enhanced verifier
            const result = await enhancedVerifiers[i].verifyAndExecute(
                proofData[i], 
                publicInputs
            );
            
            console.log(`   Step ${i} result: success=${result.success}, outputs=[${result.outputs.slice(0, 3).join(', ')}...]`);
            
        } catch (error) {
            console.log(`   Step ${i} error: ${error.message.slice(0, 100)}...`);
        }
    }
    
    // Test the sequential chain execution
    console.log("\\n⚡ Testing Sequential Chain Execution...");
    
    try {
        // Use the initial input for the first step
        const initialInput = testInputs[0];
        console.log(`   Initial input: [${initialInput.slice(0, 5).join(', ')}...] (${initialInput.length} values)`);
        console.log(`   Proof data count: ${proofData.length}`);
        
        // Estimate gas first
        console.log("   Estimating gas...");
        const gasEstimate = await sequentialExecutor.executeSequentialChain.estimateGas(
            initialInput, 
            proofData
        );
        console.log(`   Estimated gas: ${gasEstimate.toString()}`);
        
        // Execute the sequential chain
        console.log("   Executing sequential chain...");
        const tx = await sequentialExecutor.executeSequentialChain(
            initialInput, 
            proofData,
            { gasLimit: gasEstimate * 2n } // Double the estimate for safety
        );
        
        console.log("   Waiting for transaction confirmation...");
        const receipt = await tx.wait();
        
        console.log("\\n📊 Transaction Results:");
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        // Parse events
        const events = receipt.logs;
        console.log("\\n📊 Execution Events:");
        
        let stepEvents = [];
        let chainCompletedEvent = null;
        
        for (const event of events) {
            try {
                const parsedEvent = sequentialExecutor.interface.parseLog(event);
                if (parsedEvent.name === "StepExecuted") {
                    stepEvents.push(parsedEvent);
                    console.log(`   Step ${parsedEvent.args.stepIndex}: ${parsedEvent.args.stepName} - ${parsedEvent.args.success ? "✅" : "❌"}`);
                    if (parsedEvent.args.success) {
                        console.log(`      Output: [${parsedEvent.args.output.slice(0, 3).join(", ")}...] (${parsedEvent.args.output.length} values)`);
                        console.log(`      Gas: ${parsedEvent.args.gasUsed}`);
                    }
                } else if (parsedEvent.name === "ChainCompleted") {
                    chainCompletedEvent = parsedEvent;
                    console.log(`\\n🎉 Chain completed: ${parsedEvent.args.success ? "✅ SUCCESS" : "❌ FAILED"}`);
                    if (parsedEvent.args.success) {
                        console.log(`   Final output: [${parsedEvent.args.finalOutput.slice(0, 3).join(", ")}...] (${parsedEvent.args.finalOutput.length} values)`);
                    }
                    console.log(`   Total gas: ${parsedEvent.args.totalGasUsed}`);
                }
            } catch (e) {
                // Skip unparseable events
            }
        }
        
        // Summary
        console.log("\\n📈 Execution Summary:");
        console.log(`   Total steps attempted: ${stepEvents.length}`);
        console.log(`   Successful steps: ${stepEvents.filter(e => e.args.success).length}`);
        console.log(`   Failed steps: ${stepEvents.filter(e => !e.args.success).length}`);
        
        if (chainCompletedEvent) {
            console.log(`   Overall success: ${chainCompletedEvent.args.success ? "✅" : "❌"}`);
        }
        
    } catch (error) {
        console.error("\\n💥 Sequential execution failed:", error.message);
        
        // Try to get more details
        if (error.data) {
            try {
                const decodedError = sequentialExecutor.interface.parseError(error.data);
                console.error("🔍 Decoded error:", decodedError.name, decodedError.args);
            } catch (e) {
                console.error("🔍 Raw error data:", error.data);
            }
        }
        
        console.error("\\n🔧 This is expected if using dummy proof data.");
        console.error("    For real testing, use actual EZKL proof data from the model execution.");
    }
    
    // Test chain configuration
    console.log("\\n🔧 Chain Configuration Verification:");
    for (let i = 0; i < deploymentInfo.stepNames.length; i++) {
        const stepInfo = await sequentialExecutor.getStepInfo(i);
        console.log(`   Step ${i}: ${stepInfo.stepName}`);
        console.log(`      Verifier: ${stepInfo.verifierContract}`);
        console.log(`      Input size: ${stepInfo.expectedInputSize}`);
        console.log(`      Output size: ${stepInfo.expectedOutputSize}`);
    }
}

main()
    .then(() => {
        console.log("\\n✅ Sequential chain testing completed!");
        process.exit(0);
    })
    .catch((error) => {
        console.error("💥 Test failed:", error);
        process.exit(1);
    });