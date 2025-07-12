const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying Proof Composition Chain: simple_processing_chain");
    console.log("=" * 60);
    
    const chainVerifierFactory = await hre.ethers.getContractFactory("ChainVerifier");
    const chainVerifier = await chainVerifierFactory.deploy();
    await chainVerifier.deployed();
    
    console.log("📋 ChainVerifier deployed to:", chainVerifier.address);
    
    // Deploy individual verifiers and collect addresses
    const verifierAddresses = [];

    // Deploy step 00 verifier (feature_extractor)
    const step00Factory = await hre.ethers.getContractFactory("contracts/step00_feature_extractorVerifier.sol:Halo2Verifier");
    const step00Verifier = await step00Factory.deploy();
    await step00Verifier.deployed();
    console.log("   Step 00 (feature_extractor) deployed to:", step00Verifier.address);
    verifierAddresses.push(step00Verifier.address);

    // Deploy step 01 verifier (classifier)
    const step01Factory = await hre.ethers.getContractFactory("contracts/step01_classifierVerifier.sol:Halo2Verifier");
    const step01Verifier = await step01Factory.deploy();
    await step01Verifier.deployed();
    console.log("   Step 01 (classifier) deployed to:", step01Verifier.address);
    verifierAddresses.push(step01Verifier.address);

    // Deploy step 02 verifier (decision_maker)
    const step02Factory = await hre.ethers.getContractFactory("contracts/step02_decision_makerVerifier.sol:Halo2Verifier");
    const step02Verifier = await step02Factory.deploy();
    await step02Verifier.deployed();
    console.log("   Step 02 (decision_maker) deployed to:", step02Verifier.address);
    verifierAddresses.push(step02Verifier.address);

    // Set verifier addresses in the chain contract
    await chainVerifier.setVerifierAddresses(verifierAddresses);
    console.log("✅ Verifier addresses configured");
    
    // Save deployment info
    const deploymentInfo = {
        chainId: "simple_processing_chain",
        chainVerifier: chainVerifier.address,
        individualVerifiers: verifierAddresses,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("chain_deployment.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("📁 Deployment info saved to chain_deployment.json");
    
    console.log("🎉 Chain deployment completed!");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    });
