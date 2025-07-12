const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying Sequential Proof Composition Chain");
    console.log("================================================================");
    console.log("⏰ Start time:", new Date().toISOString());
    
    const [deployer] = await hre.ethers.getSigners();
    console.log("👤 Deploying with account:", deployer.address);
    
    const balance = await deployer.provider.getBalance(deployer.address);
    console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH");
    
    const network = await deployer.provider.getNetwork();
    console.log("🌐 Network:", network.name, "| Chain ID:", network.chainId);
    console.log("================================================================");
    
    // Step 1: Deploy underlying EZKL verifiers
    console.log("\n📋 Deploying underlying EZKL verifiers...");
    const underlyingVerifiers = [];
    const stepNames = [];
    const inputSizes = [];
    const outputSizes = [];
    
    // Deploy step 0: feature_extractor
    console.log("Deploying step 0 underlying verifier: feature_extractor");
    const Step0Verifier = await hre.ethers.getContractFactory("contracts/step00_feature_extractorVerifier.sol:Halo2Verifier");
    const step0Verifier = await Step0Verifier.deploy();
    await step0Verifier.waitForDeployment();
    
    const step0Address = await step0Verifier.getAddress();
    console.log("✅ Step 0 underlying verifier deployed:", step0Address);
    underlyingVerifiers.push(step0Address);
    stepNames.push("feature_extractor");
    inputSizes.push(10);
    outputSizes.push(8);
    
    // Deploy step 1: classifier
    console.log("Deploying step 1 underlying verifier: classifier");
    const Step1Verifier = await hre.ethers.getContractFactory("contracts/step01_classifierVerifier.sol:Halo2Verifier");
    const step1Verifier = await Step1Verifier.deploy();
    await step1Verifier.waitForDeployment();
    
    const step1Address = await step1Verifier.getAddress();
    console.log("✅ Step 1 underlying verifier deployed:", step1Address);
    underlyingVerifiers.push(step1Address);
    stepNames.push("classifier");
    inputSizes.push(8);
    outputSizes.push(5);
    
    // Deploy step 2: decision_maker
    console.log("Deploying step 2 underlying verifier: decision_maker");
    const Step2Verifier = await hre.ethers.getContractFactory("contracts/step02_decision_makerVerifier.sol:Halo2Verifier");
    const step2Verifier = await Step2Verifier.deploy();
    await step2Verifier.waitForDeployment();
    
    const step2Address = await step2Verifier.getAddress();
    console.log("✅ Step 2 underlying verifier deployed:", step2Address);
    underlyingVerifiers.push(step2Address);
    stepNames.push("decision_maker");
    inputSizes.push(5);
    outputSizes.push(3);
    
    // Step 2: Deploy enhanced verifiers
    console.log("\n🔧 Deploying enhanced verifiers...");
    const enhancedVerifiers = [];
    
    // Deploy enhanced feature_extractor verifier
    console.log("Deploying enhanced verifier for feature_extractor");
    const EnhancedFeature_ExtractorVerifier = await hre.ethers.getContractFactory("EnhancedFeature_ExtractorVerifier");
    const enhancedFeature_ExtractorVerifier = await EnhancedFeature_ExtractorVerifier.deploy(underlyingVerifiers[0]);
    await enhancedFeature_ExtractorVerifier.waitForDeployment();
    
    const enhancedFeature_ExtractorAddress = await enhancedFeature_ExtractorVerifier.getAddress();
    console.log("✅ Enhanced feature_extractor verifier deployed:", enhancedFeature_ExtractorAddress);
    enhancedVerifiers.push(enhancedFeature_ExtractorAddress);
    
    // Deploy enhanced classifier verifier
    console.log("Deploying enhanced verifier for classifier");
    const EnhancedClassifierVerifier = await hre.ethers.getContractFactory("EnhancedClassifierVerifier");
    const enhancedClassifierVerifier = await EnhancedClassifierVerifier.deploy(underlyingVerifiers[1]);
    await enhancedClassifierVerifier.waitForDeployment();
    
    const enhancedClassifierAddress = await enhancedClassifierVerifier.getAddress();
    console.log("✅ Enhanced classifier verifier deployed:", enhancedClassifierAddress);
    enhancedVerifiers.push(enhancedClassifierAddress);
    
    // Deploy enhanced decision_maker verifier
    console.log("Deploying enhanced verifier for decision_maker");
    const EnhancedDecision_MakerVerifier = await hre.ethers.getContractFactory("EnhancedDecision_MakerVerifier");
    const enhancedDecision_MakerVerifier = await EnhancedDecision_MakerVerifier.deploy(underlyingVerifiers[2]);
    await enhancedDecision_MakerVerifier.waitForDeployment();
    
    const enhancedDecision_MakerAddress = await enhancedDecision_MakerVerifier.getAddress();
    console.log("✅ Enhanced decision_maker verifier deployed:", enhancedDecision_MakerAddress);
    enhancedVerifiers.push(enhancedDecision_MakerAddress);
    
    // Step 3: Deploy sequential chain executor
    console.log("\n⛓️  Deploying sequential chain executor...");
    const SequentialChainExecutor = await hre.ethers.getContractFactory("SequentialChainExecutor");
    const sequentialChainExecutor = await SequentialChainExecutor.deploy();
    await sequentialChainExecutor.waitForDeployment();
    
    const executorAddress = await sequentialChainExecutor.getAddress();
    console.log("✅ Sequential chain executor deployed:", executorAddress);
    
    // Step 4: Configure the chain
    console.log("\n🔗 Configuring sequential chain...");
    const setChainTx = await sequentialChainExecutor.setChainSteps(
        enhancedVerifiers,
        stepNames,
        inputSizes,
        outputSizes
    );
    await setChainTx.wait();
    console.log("✅ Chain configured successfully");
    
    // Save deployment information
    const deploymentInfo = {
        sequentialChainExecutor: executorAddress,
        underlyingVerifiers: underlyingVerifiers,
        enhancedVerifiers: enhancedVerifiers,
        stepNames: stepNames,
        inputSizes: inputSizes,
        outputSizes: outputSizes,
        chainLength: 3,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("sequential_deployment.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("\n📁 Deployment info saved to sequential_deployment.json");
    
    // Final summary
    console.log("\n🎉 SEQUENTIAL DEPLOYMENT SUMMARY");
    console.log("================================================================");
    console.log("⛓️  Sequential Executor:", executorAddress);
    console.log("📊 Enhanced Verifiers:");
    for (let i = 0; i < enhancedVerifiers.length; i++) {
        console.log(`   ${i}: ${stepNames[i]} -> ${enhancedVerifiers[i]}`);
    }
    console.log("⏰ Completion time:", new Date().toISOString());
    console.log("🎯 Ready for sequential chain execution!");
    console.log("================================================================");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        console.error("🔍 Stack trace:", error.stack);
        process.exit(1);
    });